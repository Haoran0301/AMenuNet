import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_all_deterministic_alloc(n_agents, m_items, unit_demand = False) -> torch.tensor: # n buyers, m items -> alloc (n+1, m)
    alloc_num = (n_agents+1) ** (m_items)
    def gen(t, i, j):
        x = (n_agents+1) ** (m_items - 1 - j)
        return np.where((t // x) % (n_agents+1) == i, 1.0, 0.0)
    alloc = np.fromfunction(gen, (alloc_num-1, n_agents, m_items))
    return torch.tensor(alloc).to(torch.float32)

class Transformer2DNet(nn.Module):
    def __init__(self, d_input, d_output, n_layer, n_head, d_hidden):
        super(Transformer2DNet, self).__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.n_layer = n_layer

        d_in = d_input
        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        self.fc = nn.ModuleList()
        for i in range(n_layer):
            d_out = d_hidden if i != n_layer - 1 else d_output
            self.row_transformer.append(nn.TransformerEncoderLayer(d_in, n_head, d_hidden, batch_first=True, dropout=0))
            self.col_transformer.append(nn.TransformerEncoderLayer(d_in, n_head, d_hidden, batch_first=True, dropout=0))
            self.fc.append(nn.Sequential(
                nn.Linear(d_in + 2 * d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_out)
            ))
            d_in = d_hidden

    def forward(self, input):
        bs, n_bidder, n_item, d = input.shape
        x = input
        for i in range(self.n_layer):
            row_x = x.view(-1, n_item, d)
            row = self.row_transformer[i](row_x)
            row = row.view(bs, n_bidder, n_item, -1)

            col_x = x.permute(0, 2, 1, 3).reshape(-1, n_bidder, d)
            col = self.col_transformer[i](col_x)
            col = col.view(bs, n_item, n_bidder, -1).permute(0, 2, 1, 3)

            glo = x.view(bs, n_bidder*n_item, -1).mean(1, keepdim=True)
            glo = glo.unsqueeze(1) # (bs, 1, 1, -1)
            glo = glo.repeat(1, n_bidder, n_item, 1)

            x = torch.cat([row, col, glo], dim=-1)
            x = self.fc[i](x)
        return x

class TransformerMechanism(nn.Module):
    def __init__(self, n_bidder_type, n_item_type, d_emb, n_layer, n_head, d_hidden, menu_size, continuous_context=False, cond_prob=False):
        super(TransformerMechanism, self).__init__()
        self.d_emb = d_emb
        self.continuous_context = continuous_context
        self.pre_net = nn.Sequential(
            nn.Linear(d_emb*2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )

        d_input = d_hidden
        self.n_layer, self.n_head, self.d_hidden =  n_layer, n_head, d_hidden
        self.mechanism = Transformer2DNet(d_input, 2*menu_size+1, self.n_layer, n_head, d_hidden)
        if not continuous_context:
            self.bidder_embeddings = nn.Embedding(n_bidder_type, d_emb)
            self.item_embeddings = nn.Embedding(n_item_type, d_emb)
        self.cond_prob = cond_prob
        self.menu_size = menu_size
        self.t = 0
        self.lambdanet = nn.Sequential(
            nn.Linear(menu_size, menu_size),
            nn.ReLU(),
            nn.Linear(menu_size, menu_size)
        )

    def forward(self, batch_data, softmax_temp):
        bidder_context, item_context = batch_data
        bs, n_bidder, n_item = bidder_context.shape[0], bidder_context.shape[1], item_context.shape[1]

        if self.continuous_context:
            bidder_emb = bidder_context
            item_emb = item_context
        else:
            bidder_emb = self.bidder_embeddings(bidder_context.view(-1, n_bidder))
            item_emb = self.item_embeddings(item_context.view(-1, n_item))


        x2 = bidder_emb.unsqueeze(2).repeat(1, 1, n_item, 1)
        x3 = item_emb.unsqueeze(1).repeat(1, n_bidder, 1, 1)

        x = torch.cat([x2, x3], dim=-1)
        
        x = self.pre_net(x)

        # noise = torch.normal(0, 0.1, size=x.shape).to(x.device)
        # x = torch.cat([x, noise], dim=-1)

        mechanism = self.mechanism(x)
        allocation, b, w = \
            mechanism[:, :, :, :self.menu_size], mechanism[:, :, :, self.menu_size:2*self.menu_size], mechanism[ :, :, :, -1]
        
        alloc = F.softmax(allocation * softmax_temp, dim=1)
        alloc = alloc.permute(0, 3, 1, 2)
        alloc = alloc[:,:,:-1,:]
        # alloc bs, t, n, m

        w = w.mean(-1)
        w = torch.sigmoid(w)
        w = w[:,:-1]
        # w bs, n

        b = b.mean(-2)
        # b = allocation.mean(-2)
        b = b.mean(-2)
        b = self.lambdanet(b)
        # b bs, t

        return alloc, w, b
    
class TransformerMechanismAblationB(nn.Module):
    def __init__(self, n_bidder_type, n_item_type, d_emb, n_layer, n_head, d_hidden, menu_size, continuous_context=False, cond_prob=False):
        super(TransformerMechanismAblationB, self).__init__()
        self.d_emb = d_emb
        self.continuous_context = continuous_context
        self.pre_net = nn.Sequential(
            nn.Linear(d_emb*2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )

        d_input = d_hidden
        self.n_layer, self.n_head, self.d_hidden =  n_layer, n_head, d_hidden
        self.mechanism = Transformer2DNet(d_input, menu_size+1, self.n_layer, n_head, d_hidden)
        if not continuous_context:
            self.bidder_embeddings = nn.Embedding(n_bidder_type, d_emb)
            self.item_embeddings = nn.Embedding(n_item_type, d_emb)
        self.cond_prob = cond_prob
        self.menu_size = menu_size

    def forward(self, batch_data, softmax_temp):
        bidder_context, item_context = batch_data
        bs, n_bidder, n_item = bidder_context.shape[0], bidder_context.shape[1], item_context.shape[1]

        if self.continuous_context:
            bidder_emb = bidder_context
            item_emb = item_context
        else:
            bidder_emb = self.bidder_embeddings(bidder_context.view(-1, n_bidder))
            item_emb = self.item_embeddings(item_context.view(-1, n_item))

        x2 = bidder_emb.unsqueeze(2).repeat(1, 1, n_item, 1)
        x3 = item_emb.unsqueeze(1).repeat(1, n_bidder, 1, 1)

        x = torch.cat([x2, x3], dim=-1)
        x = self.pre_net(x)

        mechanism = self.mechanism(x)
        allocation, w = \
            mechanism[:, :, :, :self.menu_size], mechanism[ :, :, :, -1]
        
        allocation = F.softmax(allocation * softmax_temp, dim=1)
        allocation = allocation.permute(0, 3, 1, 2)
        alloc = allocation[:,:,:-1,:]
        # alloc bs, t, n, m

        w = w.mean(-1)
        w = torch.sigmoid(w)
        w = w[:,:-1]
        # w bs, n

        b = torch.zeros(bs, alloc.shape[1]).to(alloc.device)
        # b bs, t
        
        return alloc, w, b
    
class TransformerMechanismAblationW(nn.Module):
    def __init__(self, n_bidder_type, n_item_type, d_emb, n_layer, n_head, d_hidden, menu_size, continuous_context=False, cond_prob=False):
        super(TransformerMechanismAblationW, self).__init__()
        self.d_emb = d_emb
        self.continuous_context = continuous_context
        self.pre_net = nn.Sequential(
            nn.Linear(d_emb*2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )

        d_input = d_hidden
        self.n_layer, self.n_head, self.d_hidden =  n_layer, n_head, d_hidden
        self.mechanism = Transformer2DNet(d_input, 2*menu_size, self.n_layer, n_head, d_hidden)
        if not continuous_context:
            self.bidder_embeddings = nn.Embedding(n_bidder_type, d_emb)
            self.item_embeddings = nn.Embedding(n_item_type, d_emb)
        self.cond_prob = cond_prob
        self.menu_size = menu_size
        self.t = 0
        self.lambdanet = nn.Sequential(
            nn.Linear(menu_size, menu_size),
            nn.ReLU(),
            nn.Linear(menu_size, menu_size)
        )

    def forward(self, batch_data, softmax_temp):
        bidder_context, item_context = batch_data
        bs, n_bidder, n_item = bidder_context.shape[0], bidder_context.shape[1], item_context.shape[1]

        if self.continuous_context:
            bidder_emb = bidder_context
            item_emb = item_context
        else:
            bidder_emb = self.bidder_embeddings(bidder_context.view(-1, n_bidder))
            item_emb = self.item_embeddings(item_context.view(-1, n_item))


        x2 = bidder_emb.unsqueeze(2).repeat(1, 1, n_item, 1)
        x3 = item_emb.unsqueeze(1).repeat(1, n_bidder, 1, 1)

        x = torch.cat([x2, x3], dim=-1)
        
        x = self.pre_net(x)

        # noise = torch.normal(0, 0.1, size=x.shape).to(x.device)
        # x = torch.cat([x, noise], dim=-1)

        mechanism = self.mechanism(x)
        allocation, b = \
            mechanism[:, :, :, :self.menu_size], mechanism[:, :, :, self.menu_size:2*self.menu_size]
        
        allocation = F.softmax(allocation * softmax_temp, dim=1)
        allocation = allocation.permute(0, 3, 1, 2)
        alloc = allocation[:,:,:-1,:]
        # alloc bs, t, n, m

        w = torch.ones(bs, n_bidder - 1).to(alloc.device)
        # w bs, n

        b = b.mean(-2)
        b = b.mean(-2)
        b = self.lambdanet(b)
        # b bs, t

        return alloc, w, b
    
class TransformerMechanismAblationWB(nn.Module):
    def __init__(self, n_bidder_type, n_item_type, d_emb, n_layer, n_head, d_hidden, menu_size, continuous_context=False, cond_prob=False):
        super(TransformerMechanismAblationWB, self).__init__()
        self.d_emb = d_emb
        self.continuous_context = continuous_context
        self.pre_net = nn.Sequential(
            nn.Linear(d_emb*2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )

        d_input = d_hidden
        self.n_layer, self.n_head, self.d_hidden =  n_layer, n_head, d_hidden
        self.mechanism = Transformer2DNet(d_input, menu_size, self.n_layer, n_head, d_hidden)
        if not continuous_context:
            self.bidder_embeddings = nn.Embedding(n_bidder_type, d_emb)
            self.item_embeddings = nn.Embedding(n_item_type, d_emb)
        self.cond_prob = cond_prob
        self.menu_size = menu_size

    def forward(self, batch_data, softmax_temp):
        bidder_context, item_context = batch_data
        bs, n_bidder, n_item = bidder_context.shape[0], bidder_context.shape[1], item_context.shape[1]

        if self.continuous_context:
            bidder_emb = bidder_context
            item_emb = item_context
        else:
            bidder_emb = self.bidder_embeddings(bidder_context.view(-1, n_bidder))
            item_emb = self.item_embeddings(item_context.view(-1, n_item))


        x2 = bidder_emb.unsqueeze(2).repeat(1, 1, n_item, 1)
        x3 = item_emb.unsqueeze(1).repeat(1, n_bidder, 1, 1)

        x = torch.cat([x2, x3], dim=-1)
        
        x = self.pre_net(x)

        # noise = torch.normal(0, 0.1, size=x.shape).to(x.device)
        # x = torch.cat([x, noise], dim=-1)

        mechanism = self.mechanism(x)
        allocation = \
            mechanism[:, :, :, :self.menu_size]
        
        allocation = F.softmax(allocation * softmax_temp, dim=1)
        allocation = allocation.permute(0, 3, 1, 2)
        alloc = allocation[:,:,:-1,:]
        # alloc bs, t, n, m

        w = torch.ones(bs, n_bidder-1).to(alloc.device)
        # w bs, n

        b = torch.zeros(bs, alloc.shape[1]).to(alloc.device)
        # b bs, t

        return alloc, w, b
    
class TransformerMechanismAblationA(nn.Module):
    def __init__(self, n_bidder_type, n_item_type, d_emb, n_layer, n_head, d_hidden, menu_size, continuous_context=False, cond_prob=False):
        super(TransformerMechanismAblationA, self).__init__()
        self.d_emb = d_emb
        self.continuous_context = continuous_context
        self.pre_net = nn.Sequential(
            nn.Linear(d_emb*2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )

        d_input = d_hidden
        self.n_layer, self.n_head, self.d_hidden =  n_layer, n_head, d_hidden
        self.mechanism = Transformer2DNet(d_input, menu_size+1, self.n_layer, n_head, d_hidden)
        if not continuous_context:
            self.bidder_embeddings = nn.Embedding(n_bidder_type, d_emb)
            self.item_embeddings = nn.Embedding(n_item_type, d_emb)
        self.cond_prob = cond_prob
        self.menu_size = menu_size
        self.t = 0
        self.lambdanet = nn.Sequential(
            nn.Linear(menu_size, menu_size),
            nn.ReLU(),
            nn.Linear(menu_size, menu_size)
        )

    def forward(self, batch_data, softmax_temp):
        bidder_context, item_context = batch_data
        bs, n_bidder, n_item = bidder_context.shape[0], bidder_context.shape[1], item_context.shape[1]

        if self.continuous_context:
            bidder_emb = bidder_context
            item_emb = item_context
        else:
            bidder_emb = self.bidder_embeddings(bidder_context.view(-1, n_bidder))
            item_emb = self.item_embeddings(item_context.view(-1, n_item))


        x2 = bidder_emb.unsqueeze(2).repeat(1, 1, n_item, 1)
        x3 = item_emb.unsqueeze(1).repeat(1, n_bidder, 1, 1)

        x = torch.cat([x2, x3], dim=-1)
        
        x = self.pre_net(x)

        # noise = torch.normal(0, 0.1, size=x.shape).to(x.device)
        # x = torch.cat([x, noise], dim=-1)

        mechanism = self.mechanism(x)
        b, w = \
            mechanism[:, :, :, :self.menu_size], mechanism[ :, :, :, -1]
        
        allocs = generate_all_deterministic_alloc(n_bidder - 1, n_item)
        allocs = allocs.unsqueeze(0).repeat(bs, 1, 1, 1).to(b.device)

        w = w.mean(-1)
        w = torch.sigmoid(w)
        w = w[:,:-1]
        # w bs, n

        b = b.mean(-2)
        b = b.mean(-2)
        b = self.lambdanet(b)
        # b bs, t

        return allocs, w, b