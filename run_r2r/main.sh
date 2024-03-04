flag=" --exp_name exp1
      --run-type eval
      --exp-config vlnce_baselines/config/exp1.yaml
      NUM_ENVIRONMENTS 1
      "

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 run.py $flag
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port 12345 run.py $flag