import argparse
import torch
from auction import ContextualAffineMaximizerAuction
from tqdm import tqdm
from logger import get_logger
import os
from gen_values import (
    generate_sample, 
    generate_sample_linear, 
    generate_sample_traditional, 
    generate_known_D,
    generate_known_E,
    generate_known_F)

def str2bool(v):
    return v.lower() in ('true', '1') 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=2)
    parser.add_argument('--m_items', type=int, default=5)
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy', type=int, default=10)
    parser.add_argument('--menu_size', type=int, default=128)
    parser.add_argument('--continuous_context', type=str2bool, default=False)
    parser.add_argument('--const_bidder_weights', type=str2bool, default=False) # 

    parser.add_argument('--d_emb', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_hidden', type=int, default=64)
    parser.add_argument('--init_softmax_temperature', type=int, default=500)
    parser.add_argument('--alloc_softmax_temperature', type=int, default=10)

    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--train_steps', type=int, default=2000)
    parser.add_argument('--train_sample_num', type=int, default = 32768)
    parser.add_argument('--eval_freq', type=int, default=500)
    parser.add_argument('--eval_sample_num', type=int, default = 32768)
    parser.add_argument('--batch_size', type=int, default = 2048)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--lr', type=float, default = 3e-4)
    parser.add_argument('--decay_round_one', type=int, default = 3000) #
    parser.add_argument('--one_lr', type=float, default = 5e-5) #
    parser.add_argument('--decay_round_two', type=int, default = 6000) #
    parser.add_argument('--two_lr', type=float, default = 1e-5) #
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--name', type=str, default='./results')
    parser.add_argument('--data', type=int, default=2)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--test_batch_size', type=int, default = 5000)
    parser.add_argument('--ablation', type=int, default=0, help='1 for w, 2 for b, 3 for w and b, 4 for deterministic')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    file_path = f"{args.name}/{args.n_agents}_{args.m_items}_{args.menu_size}_{args.seed}_{args.ablation}"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    log_path = f"{file_path}/record.log"
    logger = get_logger(log_path)
    logger.info(args)
    DEVICE = args.device

    if args.data == 0:
        my_generate_sample = generate_sample
    elif args.data == 1:
        my_generate_sample = generate_sample_linear
    elif args.data == 2:
        my_generate_sample = generate_sample_traditional
    elif args.data == 3:
        my_generate_sample = generate_known_D
    elif args.data == 4:
        my_generate_sample = generate_known_E
    elif args.data == 5:
        my_generate_sample = generate_known_F

    model = ContextualAffineMaximizerAuction(args).to(DEVICE)
    if args.load_path != None:
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.load_path).items()})

    cur_softmax_temperature = args.init_softmax_temperature
    warm_up_init = 1e-8
    warm_up_end = args.lr
    warm_up_anneal_increase = (warm_up_end - warm_up_init) / 100
    optimizer = torch.optim.Adam(model.citransnet.parameters(), lr=warm_up_init)

    bs = args.batch_size
    num_per_train = int(args.train_sample_num / bs)
    for i in tqdm(range(args.train_steps)):
        if i == args.train_steps - 1 or (i % args.eval_freq == 0 and i>=1000): # eval
            if i == args.train_steps - 1:
                torch.save(model.state_dict(), f"{file_path}/model_{i}.pt")
            with torch.no_grad():
                test_values, test_X, test_Y = my_generate_sample(args.eval_sample_num, 
                                                                 args.n_agents, args.m_items, 
                                                                 args.dx, args.dy, DEVICE)
                revenue = torch.zeros(1).to(DEVICE)
                for num in range(int(test_values.shape[0] / bs)):
                    choice_id, _, payment, allocs, _, _, _ = model.test_time_forward(test_values[num*bs:(num+1)*bs],
                                                                            test_X[num*bs:(num+1)*bs], 
                                                                            test_Y[num*bs:(num+1)*bs])
                    revenue += payment.sum()
                revenue /= test_values.shape[0]
                logger.info(f"step {i}: revenue: {revenue}")

        train_values, train_X, train_Y = my_generate_sample(args.train_sample_num, 
                                                            args.n_agents, args.m_items, 
                                                            args.dx, args.dy, DEVICE)
        reportloss = 0
        for num in range(num_per_train): # train
            optimizer.zero_grad()
            _, _, payment, allocs = model(train_values[num*bs:(num+1)*bs], 
                                          train_X[num*bs:(num+1)*bs], 
                                          train_Y[num*bs:(num+1)*bs], 
                                          cur_softmax_temperature)
            loss = - payment.sum(0).mean()
            reportloss += loss.data
            loss.backward()
            optimizer.step()
    
        if i % 1 == 0:
            logger.info(f"step {i}: loss: {reportloss / num_per_train}")

        if i <= 100: # warm up
            for p in optimizer.param_groups:
                p['lr'] += warm_up_anneal_increase

        if i == args.decay_round_one:
            for p in optimizer.param_groups:
                p['lr'] = args.one_lr
        
        if i == args.decay_round_two:
            for p in optimizer.param_groups:
                p['lr'] = args.two_lr

    # test 
    logger.info("------------Final test------------")
    bs = args.test_batch_size
    DEVICE = 'cpu'
    test_values, test_X, test_Y = torch.load(args.test_data_path)
    test_values, test_X, test_Y = test_values.to(DEVICE), test_X.to(DEVICE), test_Y.to(DEVICE)
    test_num = test_values.shape[0]
    model = model.to('cpu')
    model.device = 'cpu'
    with torch.no_grad():
        revenue = torch.zeros(1).to(DEVICE)
        for num in tqdm(range(int(test_num / bs))):
            choice_id, _, payment, allocs, w, b, removed_choice_id = model.test_time_forward(test_values[num*bs:(num+1)*bs], 
                                                                                             test_X[num*bs:(num+1)*bs], 
                                                                                             test_Y[num*bs:(num+1)*bs])
            revenue += payment.sum()
        revenue /= test_num
    logger.info(f"Final test revenue: {revenue}")