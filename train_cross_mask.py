import os
import shutil
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import Config
from utils import Batcher, setup_seed, adapter_from_parallel
from log import Logger, highlight
from data import ChIDDataset
from model import Cross_Retriever_mask
from transformers import BertTokenizer

time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
opt_level = 'O1'
def main():
    config = Config().parser.parse_args()

    setup_seed(config.seed)
    if not os.path.exists(config.logdir):
        os.mkdir(config.logdir)
    else:
        print(highlight(f"Removing {config.logdir}"))
        shutil.rmtree(config.logdir)
        assert not os.path.exists(config.logdir)
        os.mkdir(config.logdir)

    assert config.mode == 'train'

    if config.debug:
        config.num_workers = 0

    logger = Logger(os.path.join(config.logdir, config.logfile)).get_logger()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batcher = Batcher(config=config, device=device)
    retriever =Cross_Retriever_mask(config=config, tokenizer=BertTokenizer.from_pretrained(config.model_type)).to(device)

    train_data = ChIDDataset(config.train_path, config=config)

    test_data = ChIDDataset(config.test_path, config=config)
    valid_data = ChIDDataset(config.valid_path, config=config)

    train_loader = DataLoader(train_data, config.batch_size, shuffle=True,
                              num_workers=config.num_workers, collate_fn=batcher.get_batch_cross_mask)
    test_loader = DataLoader(test_data, config.eval_batch_size, shuffle=False,
                                  num_workers=config.num_workers, collate_fn=batcher.get_batch_cross_mask)
    valid_loader = DataLoader(valid_data, config.eval_batch_size, shuffle=False,
                                   num_workers=config.num_workers, collate_fn=batcher.get_batch_cross_mask)
    eps = 1e-8
    retr_optimizer = optim.AdamW(params=retriever.parameters(), lr=config.lr_retr, eps=eps)
    retr_scheduler = get_linear_schedule_with_warmup(retr_optimizer, num_warmup_steps=config.warmup_steps_retr,
                                                     num_training_steps=len(train_loader) * config.max_epochs)
    retr_criterion = nn.CrossEntropyLoss()
    if torch.cuda.device_count() > 1:
        retriever = nn.DataParallel(retriever, device_ids=list(range(torch.cuda.device_count())))

    # -------------------- Training epochs ------------------- #
    logger.info(20 * "=" + "Config" + 20 * "=")
    for k, v in vars(config).items():
        logger.info(f'{k}: {v}')
    logger.info(
        f'retriever params: {sum(param.numel() for param in retriever.parameters() if param.requires_grad)}')
    sum_dir = os.path.join(config.logdir, config.summary)
    writer = SummaryWriter(log_dir=sum_dir)

    for epoch in range(config.max_epochs):
        logger.info(f'Begin training epoch {epoch}:')
        best_score = train(retriever=retriever,
                               train_loader=train_loader,
                               retr_criterion=retr_criterion,
                               retr_optimizer=retr_optimizer,
                               retr_scheduler=retr_scheduler,
                               batcher=batcher,
                               epoch=epoch,
                               config=config,
                               logger=logger,
                               writer=writer,
                               eval_loader=valid_loader,
                               device=device)
def train(retriever,
          train_loader,  # valid_loader, #train_loader,
          retr_criterion,
          retr_optimizer,
          retr_scheduler,
          batcher,
          epoch,
          config,
          logger,
          writer,
          eval_loader,
          device):
    eval_every = len(train_loader) // config.eval_times_per_epoch + 1
    logger.info('=' * 10 + f'Eval every {eval_every} steps!' + '=' * 10)

    retr_loss_sum = 0.
    retr_acc_sum = 0.
    best_score = 0

    for step_idx, raw_batch in enumerate(tqdm(train_loader, desc=f'Train: {epoch}')):
        retriever.train()
        labels, idiom_contents = raw_batch

        labels = torch.tensor(labels, dtype=torch.long, device=device) # [b]
        idiom_contents = torch.tensor(idiom_contents, dtype=torch.long, device=device) # [b, 7, content_len]

        idiom_logits = retriever(idiom_contents=idiom_contents) # [b, 7]

        retr_acc = torch.mean((torch.argmax(idiom_logits, dim=-1) == labels).float())
        retr_acc_sum += retr_acc.item()

        idiom_logits = torch.nn.functional.log_softmax(idiom_logits / config.temperature, dim=-1) # [b, can_num(7)]
        #print(idiom_logits)
        retriever_loss = retr_criterion(idiom_logits, labels)

        retr_optimizer.zero_grad()
        retriever_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in retriever.parameters() if p.requires_grad], config.grad_clip)
        retr_loss_sum += retriever_loss.item()
        if grad_norm >= config.grad_clip:
            logger.info(
                'WARNING: Exploding Gradients in retriever {:.2f} > {:.2f}'.format(grad_norm, config.grad_clip))
        retr_optimizer.step()
        retr_scheduler.step()

        # trick
        #if not (epoch > 1 and retr_optimizer.param_groups[0]['lr'] < config.lr_retr_min):
        #    retr_scheduler.step()

        if step_idx > 0 and (step_idx + 1) % config.print_every == 0:
            retr_lr = retr_optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(
                'retr_loss: {:.4f}, retr_lr: {:.8e}, retr_acc: {:.3f} '.format(
                    retr_loss_sum / config.print_every,
                    retr_lr,
                    retr_acc_sum / config.print_every,
                ))

            retr_loss_sum = 0.
            retr_acc_sum = 0.


        #if step_idx > 0 and (step_idx + 1) % eval_every == 0:
        #    with torch.no_grad():
        #        best_score = evaluate(data_loader=eval_loader,
        #                              retriever=retriever,
        #                              batcher=batcher,
        #                              epoch=epoch,
        #                              step_idx=step_idx,
        #                              config=config,
        #                              logger=logger,
        #                              device=device,
        #                              best_score=best_score,
        #                              retr_optimizer=retr_optimizer)
        #    # new_best_scores.append(best_score)
    with torch.no_grad():
        best_score = evaluate(data_loader=eval_loader,
                              retriever=retriever,
                              batcher=batcher,
                              epoch=epoch,
                              step_idx="end_epoch",
                              config=config,
                              logger=logger,
                              device=device,
                              best_score=best_score,
                              retr_optimizer=retr_optimizer)
    return best_score

def evaluate(data_loader,
              retriever,
              batcher,
              epoch,
              step_idx,
              config,
              logger,
              device,
              best_score,
              retr_optimizer):
    logger.info(f'Begin evaluating at epoch {epoch} and step {step_idx}:')
    retr_acc_sum = 0.
    retr_num = 0
    for step_idx, raw_batch in enumerate(tqdm(data_loader, desc=f'Eval: {epoch}')):
        retriever.eval()
        labels, idiom_contents = raw_batch

        labels = torch.tensor(labels, dtype=torch.long, device=device)  # [b]
        idiom_contents = torch.tensor(idiom_contents, dtype=torch.long, device=device)  # [b, 7, content_len]

        idiom_logits = retriever(idiom_contents=idiom_contents)  # [b, 7]

        retr_acc = torch.mean((torch.argmax(idiom_logits, dim=-1) == labels).float())
        retr_acc_sum += retr_acc.item()
        retr_num += 1
    retr_acc_final = retr_acc_sum / retr_num
    if retr_acc_final > best_score:
        best_score = retr_acc_final
        logger.info(highlight(f'NEW best results of acc: {best_score}! Save model!!!'))
        torch.save({'epoch': epoch,
                    'step_idx': step_idx,
                    'retriever': retriever.state_dict(),
                    'retr_optimizer': retr_optimizer.state_dict() if retr_optimizer is not None else None,
                    'best_acc_score': best_score},
                   os.path.join(config.logdir, f'checkpoint_acc_{epoch}_{step_idx}_best.pt'))
    return best_score
if __name__ == '__main__':
    main()