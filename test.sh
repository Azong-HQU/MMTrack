
CUDA_VISIBLE_DEVICES=0,1 python tracking/test.py \
--tracker_name mmtrack --tracker_param baseline \
--dataset_name lasot_lang --threads 6 --num_gpus 2
