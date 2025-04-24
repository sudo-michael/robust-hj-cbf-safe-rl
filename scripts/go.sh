
env=Safe-Dubins3d-NoModelMismatch-v1
# for seed in 1 2 3 4 5 ; do
#   python train/train_sac_lag.py \
#     --project_name reduce_exploration_l4dc_test --group_name lrc --lrc \
#     --config train/droq_config.py \
#     --seed ${seed} \
#     --max_steps 10000 --eval_interval 1000 --start_training 0 \ 
#     --utd_ratio 20 \
#     --env_name ${env}
#   done

for seed in 1 2 3; do
    python train/train_sac_lag.py \
        --project_name reduce_exploration_l4dc_test --group_name lrc_test --lrc \
        --config train/droq_config.py \
        --seed 1 \
        --max_steps 10000 --eval_interval 1000 --start_training 0 \
        --utd_ratio 20 \
        --env_name ${env}
done

for seed in 1 2 3; do
    python train/train_sac_lag.py \
        --project_name reduce_exploration_l4dc_test --group_name cbf_test --cbf \
        --config train/droq_config.py \
        --seed 1 \
        --max_steps 10000 --eval_interval 1000 --start_training 100 \
        --utd_ratio 20 \
        --env_name ${env}
done