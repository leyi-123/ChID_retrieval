CUDA_VISIBLE_DEVICES=6 python evaluate_cross.py \
  --test_path ../ChID/test_data_tok.json \
  --idiom_max_length 40 \
  --content_max_length 300 \
  --num_workers 16 \
  --eval_batch_size 4 \
  --retr_ckpt ./ChID_1w_cross_cls_2e-5_40_300_1217/checkpoint_acc_3_3102_best.pt \
  --logdir test_1w_cross