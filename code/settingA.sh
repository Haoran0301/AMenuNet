for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 32 --n_agents 2 --m_items 2 --seed $seed --device "cuda:2"\
    --alloc_softmax_temperature 5 --train_steps 3000 --data 0 --continuous_context true --name './results/settingA'\
    --test_data_path './data/2_2_settingA.pt'
done

for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 64 --n_agents 2 --m_items 5 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 5 --train_steps 3000 --data 0 --continuous_context true --name './results/settingA'\
    --test_data_path './data/2_5_settingA.pt'
done

for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 64 --n_agents 2 --m_items 10 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 5 --train_steps 3000 --data 0 --continuous_context true --name './results/settingA'\
    --test_data_path './data/2_10_settingA.pt'
done

for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 256 --n_agents 3 --m_items 2 --seed $seed --device "cpu"\
    --alloc_softmax_temperature 5 --train_steps 3000 --data 0 --continuous_context true --name './results/settingA'\
    --test_data_path './data/3_2_settingA.pt'
done

for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 256 --n_agents 3 --m_items 5 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 1 --train_steps 8000 --data 0 --continuous_context true --name './results/settingA'\
    --test_data_path './data/3_5_settingA.pt'
done

for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 256 --n_agents 3 --m_items 10 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 1 --train_steps 8000 --data 0 --continuous_context true --name './results/settingA'\
    --test_data_path './data/3_10_settingA.pt'
done

