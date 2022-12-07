import argparse


class Config:
    parser = argparse.ArgumentParser()
    # experiments args
    parser.add_argument('--logdir', default='log/', type=str)
    parser.add_argument('--logfile', default='log.txt', type=str)
    parser.add_argument('--print_every', default=100, type=int)
    parser.add_argument('--eval_times_per_epoch', default=-1, type=int)
    parser.add_argument('--summary', default='summary', type=str)
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'pretrain_retr'])
    parser.add_argument('--seed', default=1012, type=int)
    parser.add_argument('--num_workers', default=-1, type=int)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--debug_num', default=1000, type=int)
    parser.add_argument('--save_regular_ckpt', default=False, action='store_true')
    parser.add_argument('--eval_unseen', default=False, action='store_true')
    parser.add_argument('--eval_first', default=False, action='store_true')

    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--world_size', default=-1, type=int)
    parser.add_argument('--fp16', default=False, action='store_true')

    # data args
    parser.add_argument('--train_path', default='../ChID/train_data_1w_tok.json', type=str)
    parser.add_argument('--valid_path', default='../ChID/valid_data_tok.json', type=str)
    parser.add_argument('--test_path', default='../ChID/test_data_tok.json', type=str)
    parser.add_argument('--score_train_path', default='', type=str)
    parser.add_argument('--score_valid_path', default='', type=str)
    parser.add_argument('--score_test_path', default='', type=str)


    parser.add_argument('--model_type', default='hfl/chinese-bert-wwm-ext')
    parser.add_argument('--retr_ckpt_path', default='', type=str)

    # model args
    parser.add_argument('--idiom_max_length', default=20, type=int)
    parser.add_argument('--content_max_length', default=510, type=int)


    parser.add_argument('--share_retr', default=False, action='store_true')
    parser.add_argument('--stop_retr', default=False, action='store_true')
    parser.add_argument('--use_retr_score', default=False, action='store_true')
    parser.add_argument('--mask_word', default=False, action='store_true')
    parser.add_argument('--mask_word_rate', default=0.15, type=float)


    # optimizer args
    parser.add_argument('--dropout_prob', default=0.1, type=float)
    parser.add_argument('--temperature', default=0.2, type=float)  # 1/5
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    parser.add_argument('--lr_retr', default=3e-5, type=float)
    parser.add_argument('--lr_retr_min', default=0, type=float)
    parser.add_argument('--warmup_steps_retr', default=1000, type=int)
    parser.add_argument('--max_epochs', default=20, type=int)
    parser.add_argument('--grad_clip', default=2e2, type=float)