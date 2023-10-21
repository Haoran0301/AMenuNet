for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 16 --n_agents 3 --m_items 1 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 10 --train_steps 500 --data 3 --continuous_context false --name './results/settingD_E_F'\
    --test_data_path './data/3_1_settingD.pt'
done

for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 128 --n_agents 1 --m_items 2 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 10 --train_steps 2000 --data 4 --continuous_context false --name './results/settingD_E_F'\
    --n_layer 5 --lr 5e-5 --test_data_path './data/1_2_settingE.pt'
done

for seed in 1 2 3 4 5
do
    python3 experiments.py --menu_size 40 --n_agents 1 --m_items 2 --seed $seed --device "cuda:1"\
    --alloc_softmax_temperature 10 --train_steps 2000 --data 5 --continuous_context false --name './results/settingD_E_F'\
    --test_data_path './data/1_2_settingF.pt'
done