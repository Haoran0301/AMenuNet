for seed in 1 2 3 4 5
    do
    for ablation in 0 1 2
        do
            python3 experiments.py --ablation $ablation --seed $seed --menu_size 35 --data 1\
             --device 'cuda:1' --n_agents 5 --m_items 2 --alloc_softmax_temperature 10 \
             --train_steps 3000 --continuous_context true --name './results/ablation/5x2B/'\
             --test_data_path './data/5_2_settingB.pt'
        done
    done

for seed in 1 2 3 4 5
    do
    for ablation in 4
        do
            python3 experiments.py --ablation $ablation --seed $seed --menu_size 35 --data 1\
             --device 'cuda:1' --n_agents 5 --m_items 2 --alloc_softmax_temperature 10 \
             --train_steps 3000 --continuous_context true --name './results/ablation/5x2B/'\
             --test_data_path './data/5_2_settingB.pt'
        done
    done