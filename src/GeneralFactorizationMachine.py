import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
torch.manual_seed(42)

class FactorizationMachine(nn.Module):
    def __init__(self, n, k):
        super(FactorizationMachine, self).__init__()
        self.n = n
        self.k = k
        self.linear = nn.Linear(n, 1, bias=True)
        self.V = nn.Parameter(torch.randn(n, k), requires_grad=True)

    def forward(self, x: torch.Tensor):
        square_of_sum = torch.matmul(x, self.V).pow(2).sum(dim=1, keepdim=True)
        sum_of_square = torch.matmul(x.pow(2), self.V.pow(2)).sum(dim=1, keepdim=True)
        linear_terms = self.linear(x)

        return linear_terms + 0.5 * (square_of_sum - sum_of_square)
        
# Bayesian Personalized Ranking loss
class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, positive, negative):
        return - torch.sum(torch.log(torch.sigmoid(positive - negative)), dim=0, keepdim=True)
    

def train(model: nn.Module, dataset: Dataset, epochs=10, batch_size=32, lr=0.01, weight_decay=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = BPRLoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(DEVICE)
    model.train()
    for _ in (bar := tqdm(range(epochs))):
        for positive, negative in (epoch_bar := tqdm(dataloader, leave=False)):
            optimizer.zero_grad()
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)
            loss = criterion(model(positive), model(negative))
            loss.backward()
            optimizer.step()
            epoch_bar.set_description(f'Loss: {loss.item():.4f}')
        bar.set_description(f'Loss: {loss.item():.4f}')

    return model


def predict(model: nn.Module, dataset: Dataset, batch_size=32):
    model.to(DEVICE)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for x in tqdm(dataloader):
            x = x.to(DEVICE)
            pred = model(x).detach().cpu().numpy()
            predictions.extend(pred)
    
    return np.array(predictions).flatten()