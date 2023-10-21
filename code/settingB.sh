for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 64 --n_agents 3 --m_items 2 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 3 --train_steps 3000 --data 1 --continuous_context true --name './results/settingB'\
    --test_data_path './data/3_2_settingB.pt'
done

for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 64 --n_agents 4 --m_items 2 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 3 --train_steps 3000 --data 1 --continuous_context true --name './results/settingB'\
    --test_data_path './data/4_2_settingB.pt'
done

for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 64 --n_agents 5 --m_items 2 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 3 --train_steps 3000 --data 1 --continuous_context true --name './results/settingB'\
    --test_data_path './data/5_2_settingB.pt'
done

for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 64 --n_agents 6 --m_items 2 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 3 --train_steps 8000 --data 1 --continuous_context true --name './results/settingB'\
    --test_data_path './data/6_2_settingB.pt'
done

for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 128 --n_agents 7 --m_items 2 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 1 --train_steps 8000 --data 1 --continuous_context true --name './results/settingB'\
    --test_data_path './data/7_2_settingB.pt'
done

