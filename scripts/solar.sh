#!/bin/bash
#SBATCH -J bmm
#SBATCH --gres=gpu:1       
#SBATCH --ntasks=1
#SBATCH --time=0-20:00:00  
#SBATCH --output=./output/%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --partition=mars-lab-short
#SBATCH --nodelist=cs-venus-06   
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8

#SBATCH --mail-user=mla233@sfu.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate jaxrl5

nvidia-smi

env=Safe-Dubins3d-BadModelMismatch-v1

for seed in 1 2 3 4 5; do
    python train/train_sac_lag.py \
        --project_name reduce_exploration_l4dc --group_name lrc_1 --lrc \
        --config train/droq_config.py \
        --seed ${seed} \
        --max_steps 250000 --eval_interval 5000 --start_training 10000 \
        --utd_ratio 20 \
        --env_name ${env}
done

for seed in 1 2 3 4 5; do
    python train/train_sac_lag.py \
        --project_name reduce_exploration_l4dc --group_name cbf_1 --cbf \
        --config train/droq_config.py \
        --seed ${seed} \
        --max_steps 250000 --eval_interval 5000 --start_training 10000 \
        --utd_ratio 20 \
        --env_name ${env}
done