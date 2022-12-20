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
from model import Cross_Retriever
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
    retriever =Cross_Retriever(config=config, tokenizer=BertTokenizer.from_pretrained(config.model_type)).to(device)


    test_data = ChIDDataset(config.test_path, config=config)
    test_loader = DataLoader(test_data, config.eval_batch_size, shuffle=False,
                                  num_workers=config.num_workers, collate_fn=batcher.get_batch_cross)
    eps = 1e-8
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
    with torch.no_grad():
        best_score = evaluate(data_loader=test_loader,
                              retriever=retriever,
                              batcher=batcher,
                              epoch=-1,
                              step_idx="end_epoch",
                              config=config,
                              logger=logger,
                              device=device,
                              best_score=0)

def evaluate(data_loader,
              retriever,
              batcher,
              epoch,
              step_idx,
              config,
              logger,
              device,
              best_score):
    retr_acc_sum = 0.
    retr_num = 0
    if not config.zero_shot:
        logger.info(f'Begin evaluating at epoch {epoch} and step {step_idx}:')
        retr_ckpt = torch.load(config.retr_ckpt)
        logger.info(
            highlight(
                f"Reload retriever ckpt from {config.retr_ckpt}"))
        retriever.load_state_dict(adapter_from_parallel(retr_ckpt['retriever']))
    for step_idx, raw_batch in enumerate(tqdm(data_loader, desc=f'Eval: {epoch}')):
        retriever.eval()
        labels, mask_locations, idiom_contents = raw_batch

        labels = torch.tensor(labels, dtype=torch.long, device=device)  # [b]
        mask_locations = torch.tensor(mask_locations, dtype=torch.long, device=device)  # [b]
        idiom_contents = torch.tensor(idiom_contents, dtype=torch.long, device=device)  # [b, 7, content_len]

        idiom_logits = retriever(idiom_contents=idiom_contents, mask_locations=mask_locations)  # [b, 7]

        retr_acc = torch.mean((torch.argmax(idiom_logits, dim=-1) == labels).float())
        retr_acc_sum += retr_acc.item()
        retr_num += 1
    retr_acc_final = retr_acc_sum / retr_num
    logger.info(highlight(f'NEW best results of acc: {retr_acc_final} for test set!'))
    return best_score
if __name__ == '__main__':
    main()