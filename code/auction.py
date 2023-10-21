import numpy as np
import torch
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
from net import (TransformerMechanism, 
                 TransformerMechanismAblationB, 
                 TransformerMechanismAblationW, 
                 TransformerMechanismAblationWB,
                 TransformerMechanismAblationA)
from IPython import embed

def generate_all_deterministic_alloc(n_agents, m_items, unit_demand = False) -> torch.tensor: # n buyers, m items -> alloc (n+1, m)
    alloc_num = (n_agents+1) ** (m_items)
    def gen(t, i, j):
        x = (n_agents+1) ** (m_items - 1 - j)
        return np.where((t // x) % (n_agents+1) == i, 1.0, 0.0)
    alloc = np.fromfunction(gen, (alloc_num-1, n_agents, m_items))
    return torch.tensor(alloc).to(torch.float32)

class ContextualAffineMaximizerAuction(nn.Module):
    def __init__(self, args, oos=False) -> None:
        super().__init__()
        self.n_agents = args.n_agents
        self.m_items = args.m_items
        self.dx = args.dx
        self.dy = args.dy
        self.device = args.device
        self.menu_size = args.menu_size
        self.continuous = args.continuous_context
        self.const_bidder_weights = args.const_bidder_weights
        self.alloc_softmax_temperature = args.alloc_softmax_temperature

        mask = 1 - torch.eye((self.n_agents)).to(self.device)
        self.mask = torch.zeros(args.n_agents, args.batch_size, args.n_agents).to(self.device)
        for i in range(args.n_agents):
            self.mask[i] = mask[i].repeat(args.batch_size, 1)
        self.mask = self.mask.reshape(args.n_agents * args.batch_size, args.n_agents)

        # for ablation study
        if args.ablation == 0:
            self.citransnet = TransformerMechanism(self.n_agents + 1, self.m_items, args.d_emb, args.n_layer, args.n_head, args.d_hidden, 
                args.menu_size, continuous_context=args.continuous_context, cond_prob=False).to(self.device)
        elif args.ablation == 1:
            self.citransnet = TransformerMechanismAblationW(self.n_agents + 1, self.m_items, args.d_emb, args.n_layer, args.n_head, args.d_hidden, 
                args.menu_size, continuous_context=args.continuous_context, cond_prob=False).to(self.device)
        elif args.ablation == 2:
            self.citransnet = TransformerMechanismAblationB(self.n_agents + 1, self.m_items, args.d_emb, args.n_layer, args.n_head, args.d_hidden, 
                args.menu_size, continuous_context=args.continuous_context, cond_prob=False).to(self.device)
        elif args.ablation == 3:
            self.citransnet = TransformerMechanismAblationWB(self.n_agents + 1, self.m_items, args.d_emb, args.n_layer, args.n_head, args.d_hidden, 
                args.menu_size, continuous_context=args.continuous_context, cond_prob=False).to(self.device)
        elif args.ablation == 4:
            self.citransnet = TransformerMechanismAblationA(self.n_agents + 1, self.m_items, args.d_emb, args.n_layer, args.n_head, args.d_hidden, 
                args.menu_size, continuous_context=args.continuous_context, cond_prob=False).to(self.device)
        else:
            raise ValueError("ablation number invalid")
        # V, X, Y -> alloc, mu, lambda

    def test_time_forward(self, input_bids: torch.tensor, X, Y) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''
        input_bids: B, n, m 
        X: B, n_agents, dx 
        Y: B, m_items, dy
        '''
        B, n, m = input_bids.shape
        if self.continuous:
            X = torch.cat((X, torch.ones(B, 1, self.dx).to(self.device)), axis=1)
        else:
            X = torch.cat((X, torch.ones(B, 1).to(self.device).long() * self.n_agents), axis=1)

        allocs, w, b = self.citransnet((X, Y), self.alloc_softmax_temperature)

        if self.const_bidder_weights == True:
            w = torch.ones(B, n).to(self.device)
        allocs = torch.cat((allocs, torch.zeros(B, 1, n, m).to(self.device)), 1) # B, t, n, m
        b = torch.cat((b, torch.zeros((B, 1)).to(self.device)), 1) # B, t
        assert w.all() > 0

        util_from_items = (allocs * input_bids.unsqueeze(1)).sum(axis=-1) # B, t, n
        per_agent_welfare = w.unsqueeze(1) * util_from_items # B, t, n
        total_welfare = per_agent_welfare.sum(axis=-1) # B, t
        alloc_choice_ind = torch.argmax(total_welfare + b, -1)  # B

        item_allocation = [allocs[i, alloc_choice_ind[i],...] for i in range(B)] 
        item_allocation = torch.stack(item_allocation) # B, n, m
        chosen_alloc_welfare_per_agent = [per_agent_welfare[i, alloc_choice_ind[i], ...] for i in range(B)]
        chosen_alloc_welfare_per_agent = torch.stack(chosen_alloc_welfare_per_agent) # B, n
        
        ####
        removed_alloc_choice_ind_list = []
        ####

        payments = []
        if self.n_agents > 1:
            for i in range(self.n_agents):
                mask = torch.ones(n).to(self.device)
                mask[i] = 0
                removed_i_welfare = per_agent_welfare * mask.reshape(1, 1, n)
                total_removed_welfare = removed_i_welfare.sum(-1) # B, t
                removed_alloc_choice_ind = torch.argmax(total_removed_welfare + b, -1) # B
                removed_chosen_welfare = [total_removed_welfare[i, removed_alloc_choice_ind[i]] for i in range(B)] # B
                removed_chosen_welfare = torch.stack(removed_chosen_welfare)
                
                removed_alloc_b = [b[i, removed_alloc_choice_ind[i]] for i in range(B)]
                removed_alloc_b = torch.stack(removed_alloc_b)

                alloc_b = [b[i, alloc_choice_ind[i]] for i in range(B)]
                alloc_b = torch.stack(alloc_b)

                payments.append(
                    (1.0 / w[:,i])
                    * (
                        (
                            removed_chosen_welfare
                            + removed_alloc_b
                        )
                        - (chosen_alloc_welfare_per_agent.sum(1) - chosen_alloc_welfare_per_agent[:,i])
                        - alloc_b
                    )
                )
                removed_alloc_choice_ind_list.append(removed_alloc_choice_ind)
        else:
            payments = [-b[i,alloc_choice_ind[i]] for i in range(B)] # special case for 1 agent -- just charge boost

        payments = torch.stack(payments)
        return alloc_choice_ind, item_allocation, payments, allocs, w, b, removed_alloc_choice_ind_list
    
    def forward(self, input_bids: torch.tensor, X, Y, softmax_temp: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''
        input_bids: B, n, m 
        X: B, n_agents, dx 
        Y: B, m_items, dy
        '''
        B, n, m = input_bids.shape
        if self.continuous:
            X = torch.cat((X, torch.ones(B, 1, self.dx).to(self.device)), axis=1)
        else:
            X = torch.cat((X, torch.ones(B, 1).to(self.device).long()* self.n_agents), axis=1)

        allocs, w, b = self.citransnet((X, Y), self.alloc_softmax_temperature)
        if self.const_bidder_weights == True:
            w = torch.ones(B, n).to(self.device)

        allocs = torch.cat((allocs, torch.zeros(B, 1, n, m).to(self.device)), 1) # B, t, n, m
        b = torch.cat((b, torch.zeros((B, 1)).to(self.device)), 1) # B, t

        util_from_items = (allocs * input_bids.unsqueeze(1)).sum(axis=-1) # B, t, n
        per_agent_welfare = w.unsqueeze(1) * util_from_items # B, t, n
        total_welfare = per_agent_welfare.sum(axis=-1) # B, t
        alloc_choice = F.softmax((total_welfare + b) * softmax_temp, dim=-1) # B, t
        item_allocation = (torch.unsqueeze(torch.unsqueeze(alloc_choice, -1), -1) * allocs).sum(axis=1)
        chosen_alloc_welfare_per_agent = (per_agent_welfare * torch.unsqueeze(alloc_choice, -1)).sum(axis=1) # B, n 

        if n > 1:
            n_chosen_alloc_welfare_per_agent= chosen_alloc_welfare_per_agent.repeat(n, 1)# nB, n
            masked_chosen_alloc_welfare_per_agent = n_chosen_alloc_welfare_per_agent * self.mask #  nB, n
            n_per_agent_welfare = per_agent_welfare.repeat(n, 1, 1)# nB, t, n
            removed_i_welfare = n_per_agent_welfare * self.mask.reshape(n*B, 1, n) # nB, t, n
            total_removed_welfare  = removed_i_welfare.sum(axis=-1) # nB, t
            removed_alloc_choice = F.softmax((total_removed_welfare + b.repeat(n, 1)) * softmax_temp, dim=-1)
            # nB, t
            removed_chosen_welfare_per_agent = (
                removed_i_welfare * removed_alloc_choice.unsqueeze(-1) # nB, t, n
            ).sum(axis=1)
            # nB, n
            payments = torch.zeros(n * B).to(self.device)
            payments = (1 / w.permute(1, 0).reshape(n * B)) * (
                removed_chosen_welfare_per_agent.sum(-1)
                + (removed_alloc_choice * b.repeat(n, 1)).sum(-1)
                - masked_chosen_alloc_welfare_per_agent.sum(-1)
                - (alloc_choice * b).sum(1).repeat(n)
            ) # nB
            payments = payments.reshape(n, B)
        else:
            payments = - (alloc_choice * b).sum(1).reshape(1, B)# special case for 1 agent -- just charge boost
        return alloc_choice, item_allocation, payments, allocs
    


def VCG(input_bids, device):
    B, n, m = input_bids.shape

    allocs = generate_all_deterministic_alloc(n, m)
    allocs = allocs.unsqueeze(0)

    util_from_items = (allocs * input_bids.unsqueeze(1)).sum(axis=-1) # B, t, n
    per_agent_welfare = util_from_items # B, t, n
    total_welfare = per_agent_welfare.sum(axis=-1) # B, t
    alloc_choice_ind = torch.argmax(total_welfare, -1)  # B

    item_allocation = [allocs[0, alloc_choice_ind[i],...] for i in range(B)] 
    item_allocation = torch.stack(item_allocation) # B, n, m
    chosen_alloc_welfare_per_agent = [per_agent_welfare[i, alloc_choice_ind[i], ...] for i in range(B)]
    chosen_alloc_welfare_per_agent = torch.stack(chosen_alloc_welfare_per_agent) # B, n
    
    payments = []
    if n > 1:
        for i in range(n):
            mask = torch.ones(n).to(device)
            mask[i] = 0
            removed_i_welfare = per_agent_welfare * mask.reshape(1, 1, n)
            total_removed_welfare = removed_i_welfare.sum(-1) # B, t
            removed_alloc_choice_ind = torch.argmax(total_removed_welfare, -1) # B
            removed_chosen_welfare = [total_removed_welfare[i, removed_alloc_choice_ind[i]] for i in range(B)] # B
            removed_chosen_welfare = torch.stack(removed_chosen_welfare)

            payments.append(
                (
                    (
                        removed_chosen_welfare
                    )
                    - (chosen_alloc_welfare_per_agent.sum(1) - chosen_alloc_welfare_per_agent[:,i])
                )
            )
    else:
        payments = 0

    payments = torch.stack(payments)
    return alloc_choice_ind, item_allocation, payments, allocs