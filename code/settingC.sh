for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 64 --n_agents 2 --m_items 5 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 10 --train_steps 3000 --data 2 --continuous_context false --name './results/settingC'\
    --test_data_path './data/2_5_settingC.pt'
done

# for seed in 1 2 3 4 5
# do
#     python3 experiments.py --menu_size 512 --n_agents 3 --m_items 5 --seed $seed --device "cuda:3"\
#     --alloc_softmax_temperature 10 --train_steps 2000 --data 2 --continuous_context false --name './results/settingC'\
#     --test_data_path './data/3_5_settingC.pt'
# done

# for seed in 1 2 3 4 5
# do
#     python3 experiments.py --menu_size 1024 --n_agents 5 --m_items 5 --seed $seed --device "cuda:1"\
#     --alloc_softmax_temperature 50 --train_steps 2000 --data 2 --continuous_context false --name './results/settingC'\
#     --n_layer 5 --test_data_path './data/5_5_settingC.pt'
# done
