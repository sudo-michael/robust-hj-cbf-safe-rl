#!/bin/bash
#SBATCH -J drop_q
#SBATCH --gres=gpu:1       
#SBATCH --ntasks=1
#SBATCH --time=0-04:30:00  
#SBATCH --output=./output/%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --partition=debug   # debug: max 5 hours   short: 2 days long: 7 days
#SBATCH --nodelist=cs-venus-09   
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

python examples/train_sac_lag.py \
    --config examples/droq_config.py \
    --env_name=Safe-StaticDubins3d-v1 \
    --cbf_gamma=1.0 \
    --utd_ratio=$UTD_RATIO \
    # --critic_dropout_rate=$CDR \
    # --critic_layer_norm=$CLN \
    # --init_temperature=$IT \
    --max_steps=250000
    --seed=$SEED
    --group_name=$GN
