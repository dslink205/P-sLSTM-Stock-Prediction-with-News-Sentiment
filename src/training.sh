#!/bin/bash

export PYTORCH_NO_BF16=1  # 禁用 bf16
export TORCH_CUDA_ARCH_LIST="7.5"  # 限制 T4 架构

python /mnt/data/wqk/AI+Fin/models/P-sLSTM/run_longExp.py \
  --data AAPL_with_sentiment \
  --root_path /mnt/data/wqk/AI+Fin/data/merge_pro/ \
  --data_path 105.AAPL_苹果_with_sentiment_processed.csv \
  --features MS \
  --target label \
  --freq d \
  --seq_len 30 \
  --label_len 15 \
  --pred_len 1 \
  --is_training 1 \
  --c_out 1 \
  --embedding_dim 64 \
  --model P_sLSTM
