for seed in 1 2 3 4 5
    do
    for ablation in 0 1 2
        do
            python3 experiments.py --ablation $ablation --seed $seed --menu_size 64 --data 0\
             --device 'cuda:2' --n_agents 2 --m_items 10 --alloc_softmax_temperature 10 \
             --train_steps 1000 --continuous_context true --name './results/ablation/2x10A/'\
             --test_data_path './data/2_10_settingA.pt'
        done
    done