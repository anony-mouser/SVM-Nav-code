export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag=" --exp_name exp_mp_1839_0.5value_map
      --run-type eval
      --exp-config vlnce_baselines/config/exp1.yaml
      --nprocesses 12
      NUM_ENVIRONMENTS 1
      TRAINER_NAME ZS-Evaluator-mp
      TORCH_GPU_IDS [0,1,2,3,4,5]
      SIMULATOR_GPU_IDS [0,1,2,3,4,5]
      "
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python run_mp.py $flag

# flag=" --exp_name exp_1_choice_check
#       --run-type eval
#       --exp-config vlnce_baselines/config/exp1.yaml
#       --nprocesses 1
#       NUM_ENVIRONMENTS 1
#       TRAINER_NAME ZS-Evaluator-mp
#       TORCH_GPU_IDS [0]
#       SIMULATOR_GPU_IDS [0]
#       "
# CUDA_VISIBLE_DEVICES=2 python run_mp.py $flag