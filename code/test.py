import argparse
import torch
from auction import ContextualAffineMaximizerAuction, VCG
from tqdm import tqdm
import numpy as np

def str2bool(v):
    return v.lower() in ('true', '1') 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=2)
    parser.add_argument('--m_items', type=int, default=2)
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy', type=int, default=10)

    parser.add_argument('--d_emb', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_hidden', type=int, default=64)

    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--continuous_context', type=str2bool, default=False)
    parser.add_argument('--menu_size', type=int, default=1024)
    parser.add_argument('--const_bidder_weights', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--alloc_softmax_temperature', type=int, default=5, help='tau_A')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--VCG', type=str2bool, default=False)
    parser.add_argument('--ablation', type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    DEVICE = args.device
    VCG_test= args.VCG

    torch.manual_seed(2002)
    if not VCG_test:
        model = ContextualAffineMaximizerAuction(args).to(DEVICE)
        if args.load_path != None:
            model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.load_path).items()})
            
    bs = args.batch_size
    test_values, test_X, test_Y = torch.load(args.data_path)
    test_values, test_X, test_Y = test_values.to(DEVICE), test_X.to(DEVICE), test_Y.to(DEVICE)

    test_num = test_values.shape[0]
    # choice_id_record = torch.zeros(args.menu_size + 1)
    # removed_choice_id_record = torch.zeros(args.menu_size + 1)
    # randomize_cnt = 0
    # tot_cnt = 0
    with torch.no_grad():
        revenue = torch.zeros(1).to(DEVICE)
        for num in tqdm(range(int(test_num / bs))):
            if not VCG_test:
                choice_id, _, payment, allocs, w, b, removed_choice_id = model.test_time_forward(test_values[num*bs:(num+1)*bs], test_X[num*bs:(num+1)*bs], test_Y[num*bs:(num+1)*bs])
            else:
                choice_id, _, payment, allocs = VCG(test_values[num*bs:(num+1)*bs], DEVICE)
            revenue += payment.sum()
            # for i in range(choice_id.shape[0]):
            #     flag = 0
            #     x = allocs[i, choice_id[i]]
            #     x = x[x>0.01]
            #     x = x[x<0.99]
            #     if x.shape[0] > 0:
            #         randomize_cnt += 1
            #         flag = 1
            #     choice_id_record[choice_id[i]] += 1
            #     for j in range(args.n_agents):
            #         x = allocs[i, removed_choice_id[j][i]]
            #         x = x[x>0.01]
            #         x = x[x<0.99]
            #         if x.shape[0] > 0:
            #             randomize_cnt += 1
            #             flag = 1
            #         removed_choice_id_record[removed_choice_id[j][i]] += 1
            #     if flag == 1:
            #         tot_cnt += 1
        revenue /= test_num
    print(revenue)
    # print(randomize_cnt)
    # print(tot_cnt)
    # torch.save((choice_id_record, removed_choice_id_record, allocs, w, b), 
    #            f'{args.n_agents}_{args.m_items}_{args.menu_size}_record.pt')
