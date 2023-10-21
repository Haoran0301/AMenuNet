for seed in 1 2 3 4 5
    do
    for ablation in 0 1 2 4
        do
            python3 experiments.py --ablation $ablation --seed $seed --menu_size 26 --data 2\
             --device 'cuda:3' --n_agents 2 --m_items 3 --alloc_softmax_temperature 10 \
             --train_steps 1000 --continuous_context false --name './results/ablation/2x3C/'\
             --test_data_path './data/2_3_settingC.pt'
        done
    done