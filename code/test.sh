# for seed in 1 2 3 4 5
#     do
#     python3 test.py --n_agents 2 --m_items 2 --menu_size 32 --data_path './data/2_2_settingA.pt'\
#     --load_path "./results/settingA/2_2_32_${seed}_0/model_2999.pt" --alloc_softmax_temperature 5 --continuous_context True

#     python3 test.py --n_agents 2 --m_items 5 --menu_size 64 --data_path './data/2_5_settingA.pt'\
#     --load_path "./results/settingA/2_5_64_${seed}_0/model_2999.pt" --alloc_softmax_temperature 5 --continuous_context True

#     python3 test.py --n_agents 2 --m_items 10 --menu_size 64 --data_path './data/2_10_settingA.pt'\
#     --load_path "./results/settingA/2_10_64_${seed}_0/model_2999.pt" --alloc_softmax_temperature 5 --continuous_context True

#     python3 test.py --n_agents 3 --m_items 2 --menu_size 64 --data_path './data/3_2_settingA.pt'\
#     --load_path "./results/settingA/3_2_64_${seed}_0/model_2999.pt" --alloc_softmax_temperature 5 --continuous_context True

#     python3 test.py --n_agents 3 --m_items 5 --menu_size 256 --data_path './data/3_5_settingA.pt'\
#     --load_path "./results/settingA/3_5_256_${seed}_0/model_7999.pt" --alloc_softmax_temperature 1 --continuous_context True

#     python3 test.py --n_agents 3 --m_items 10 --menu_size 256 --data_path './data/3_10_settingA.pt'\
#     --load_path "./results/settingA/3_10_256_${seed}_0/model_7999.pt" --alloc_softmax_temperature 1 --continuous_context True

#     python3 test.py --n_agents 3 --m_items 2 --menu_size 64 --data_path './data/3_2_settingB.pt'\
#     --load_path "./results/settingB/3_2_64_${seed}_0/model_2999.pt" --alloc_softmax_temperature 3 --continuous_context True

#     python3 test.py --n_agents 4 --m_items 2 --menu_size 64 --data_path './data/4_2_settingB.pt'\
#     --load_path "./results/settingB/4_2_64_${seed}_0/model_2999.pt" --alloc_softmax_temperature 3 --continuous_context True

#     python3 test.py --n_agents 5 --m_items 2 --menu_size 64 --data_path './data/5_2_settingB.pt'\
#     --load_path "./results/settingB/5_2_64_${seed}_0/model_2999.pt" --alloc_softmax_temperature 3 --continuous_context True

#     python3 test.py --n_agents 6 --m_items 2 --menu_size 64 --data_path './data/6_2_settingB.pt'\
#     --load_path "./results/settingB/6_2_64_${seed}_0/model_7999.pt" --alloc_softmax_temperature 3 --continuous_context True

#     python3 test.py --n_agents 7 --m_items 2 --menu_size 128 --data_path './data/7_2_settingB.pt'\
#     --load_path "./results/settingB/7_2_128_${seed}_0/model_7999.pt" --alloc_softmax_temperature 1 --continuous_context True

#     python3 test.py --n_agents 2 --m_items 5 --menu_size 128 --data_path './data/2_5_settingC.pt'\
#     --load_path "./results/settingC/2_5_128_${seed}_0/model_2999.pt" --alloc_softmax_temperature 10 --continuous_context False

#     python3 test.py --n_agents 3 --m_items 10 --menu_size 512 --data_path './data/3_10_settingC.pt'\
#     --load_path "./results/settingC/3_10_512_${seed}_0/model_1999.pt" --alloc_softmax_temperature 10 --continuous_context False

#     python3 test.py --n_agents 5 --m_items 5 --menu_size 1024 --data_path './data/5_5_settingC.pt' --n_layer 5\
#     --load_path "./results/settingC/5_5_1024_${seed}_0/model_1999.pt" --alloc_softmax_temperature 50 --continuous_context False

#     python3 test.py --n_agents 3 --m_items 1 --menu_size 16 --data_path './data/3_1_settingD.pt'\
#     --load_path "./results/settingD_E_F/3_1_16_${seed}_0/model_499.pt" --alloc_softmax_temperature 10 --continuous_context False

#     python3 test.py --n_agents 1 --m_items 2 --menu_size 128 --data_path './data/1_2_settingE.pt' --n_layer 5\
#     --load_path "./results/settingD_E_F/1_2_128_${seed}_0/model_999.pt" --alloc_softmax_temperature 10 --continuous_context False

#     python3 test.py --n_agents 1 --m_items 2 --menu_size 40 --data_path './data/1_2_settingF.pt'\
#     --load_path "./results/settingD_E_F/1_2_40_${seed}_0/model_1999.pt" --alloc_softmax_temperature 10 --continuous_context False
#     done

# oos example
# for item in 2 3 4 5 6 7 8 9 10
#     do
#     python3 test.py --n_agents 2 --m_items $item --menu_size 32 --data_path "./data/2_${item}_settingA.pt"\
#     --load_path "./results/settingA/2_2_32_1_0/model_2999.pt" --alloc_softmax_temperature 5 --continuous_context True
#     done

# for item in 2 3 4 5 6 7 8 9 10
#     do
#     python3 test.py --n_agents 2 --m_items $item --menu_size 64 --data_path "./data/2_${item}_settingA.pt"\
#     --load_path "./results/settingA/2_5_64_1_0/model_2999.pt" --alloc_softmax_temperature 5 --continuous_context True
#     done