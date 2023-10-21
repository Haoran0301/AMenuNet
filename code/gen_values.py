import numpy as np
import torch

def generate_sample(sample_num, n_agents, m_items, dx, dy, device):
    X = torch.rand((sample_num, n_agents, dx))
    X = X * 2 - 1
    Y = torch.rand((sample_num, m_items, dy))
    Y = Y * 2 - 1
    upper = np.fromfunction(lambda t, i, j: torch.sigmoid((X[t, i] * Y[t, j]).sum(-1)), (sample_num, n_agents, m_items)).clone().detach()
    values = torch.rand((sample_num, n_agents, m_items)) * upper
    return values.to(device), X.to(device), Y.to(device)

def generate_sample_linear(sample_num, n_agents, m_items, dx, dy, device): #只适用于item = 2
    X = torch.rand((sample_num, n_agents, dx)) * 2 - 1
    Y = torch.rand((sample_num, m_items, dy)) * 2 - 1
    upper = np.fromfunction(lambda t, i, j: torch.sigmoid((X[t, i] * Y[t, j]).sum(-1)), (sample_num, n_agents, m_items)).clone().detach() 
    values = torch.rand((sample_num, n_agents, 1))
    values = torch.cat((values, 1 - values), dim=-1)
    values = values * upper
    return values.to(device), X.to(device), Y.to(device)

def generate_sample_traditional(sample_num, n_agents, m_items, dx, dy, device):
    values = torch.rand((sample_num, n_agents, m_items))
    X = torch.arange(n_agents).repeat(sample_num).reshape(sample_num, n_agents).long()
    Y = torch.arange(m_items).repeat(sample_num).reshape(sample_num, m_items).long()
    return values.to(device), X.to(device), Y.to(device)


def generate_known_D(sample_num, n_agents, m_items, dx, dy, device):
    values = np.random.exponential(3.0, size=(sample_num, 3, 1))
    values = torch.tensor(values)
    X = torch.arange(3).repeat(sample_num).reshape(sample_num, 3).long()
    Y = torch.arange(1).repeat(sample_num).reshape(sample_num, 1).long()
    return values.to(device), X.to(device), Y.to(device)

def generate_known_E(sample_num, n_agents, m_items, dx, dy, device):
    values1 = torch.rand((sample_num, 1, 1)) * 12 + 4
    values2 = torch.rand((sample_num, 1, 1)) * 3 + 4
    values = torch.cat((values1, values2), -1)
    X = torch.arange(1).repeat(sample_num).reshape(sample_num, 1).long()
    Y = torch.arange(2).repeat(sample_num).reshape(sample_num, 2).long()
    return values.to(device), X.to(device), Y.to(device)

def generate_known_F(sample_num, n_agents, m_items, dx, dy, device):
    values1 = torch.rand((sample_num, 1, 1))
    values1 = (1 / (1 - values1)) ** 0.2 - 1
    values2 = torch.rand((sample_num, 1, 1))
    values2 = (1 / (1 - values2)) ** (1/6) - 1
    values = torch.cat((values1, values2), -1)
    X = torch.arange(1).repeat(sample_num).reshape(sample_num, 1).long()
    Y = torch.arange(2).repeat(sample_num).reshape(sample_num, 2).long()
    return values.to(device), X.to(device), Y.to(device)




sample_num = 100000
torch.manual_seed(2002)
for bidder in [2, 3]:
    for item in range(2, 11):
        x = generate_sample(sample_num, bidder, item, 10, 10, 'cpu')
        torch.save(x, f'./data/{bidder}_{item}_settingA.pt')

for bidder in range(2, 11):
    for item in [2]:
        x = generate_sample_linear(sample_num, bidder, item, 10, 10, 'cpu')
        torch.save(x, f'./data/{bidder}_{item}_settingB.pt')

bidder, item = 2, 5
x = generate_sample_traditional(sample_num, bidder, item, 10, 10, 'cpu')
torch.save(x, f'./data/{bidder}_{item}_settingC.pt')

bidder = 3
for item in range(2, 11):
    x = generate_sample_traditional(sample_num, bidder, item, 10, 10, 'cpu')
    torch.save(x, f'./data/{bidder}_{item}_settingC.pt')

bidder, item = 5, 5
x = generate_sample_traditional(sample_num, bidder, item, 10, 10, 'cpu')
torch.save(x, f'./data/{bidder}_{item}_settingC.pt')

bidder, item = 3, 1
x = generate_known_D(sample_num, bidder, item, 10, 10, 'cpu')
torch.save(x, f'./data/{bidder}_{item}_settingD.pt')

bidder, item = 1, 2
x = generate_known_E(sample_num, bidder, item, 10, 10, 'cpu')
torch.save(x, f'./data/{bidder}_{item}_settingE.pt')

bidder, item = 1, 2
x = generate_known_F(sample_num, bidder, item, 10, 10, 'cpu')
torch.save(x, f'./data/{bidder}_{item}_settingF.pt')