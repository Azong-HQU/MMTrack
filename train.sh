
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py \
--script mmtrack --config baseline \
--save_dir ./output \
--mode multiple --nproc_per_node 2 \
--use_wandb 0