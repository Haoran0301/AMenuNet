for seed in 1 2 3 4 5
    do
    for ablation in 1 2
        do
            python3 experiments.py --ablation $ablation --seed $seed --menu_size 512 --data 2\
             --device 'cuda:3' --n_agents 3 --m_items 10 --n_layer 3 --alloc_softmax_temperature 10 \
             --train_steps 2000 --continuous_context false --name './results/ablation/3x10C/'\
             --test_data_path './data/3_10_settingC.pt'
        done
    done
