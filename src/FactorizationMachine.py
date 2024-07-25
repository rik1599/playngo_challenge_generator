import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# The model
class TwoWayFM(nn.Module):
    def __init__(self, num_features, embedding_size) -> None:
        super(TwoWayFM, self).__init__()

        self.w0 = nn.Linear(1, 1, bias=True)
        self.w = nn.Embedding(num_features, 1)
        self.V = nn.Embedding(num_features, embedding_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        square_of_sum = torch.sum(self.V(x), dim=1) ** 2
        sum_of_square = torch.sum(self.V(x) ** 2, dim=1)
        x = self.w0(torch.sum(self.w(x), dim=1)) + 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)
        return self.sigmoid(x)
    

class TwoWayFMDataset(Dataset):
    def __init__(self, X: np.ndarray, features_dims, y: np.ndarray = None) -> None:
        self.y = y
        self.X = X

        # Embedding matrix offsets
        self.offsets = [0] + np.cumsum(features_dims).tolist()[:-1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = [x + offset for x, offset in zip(self.X[idx], self.offsets)]

        if self.y is None:
            return torch.tensor(X, dtype=torch.long)
        else:
            return torch.tensor(X, dtype=torch.long), torch.tensor([self.y[idx]], dtype=torch.float)


def fit(model: nn.Module, dataset: Dataset, epochs: int, batch_size: int):
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    criterion = nn.BCELoss()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} loss: {loss.item()}')


def predict(model: nn.Module, dataset: Dataset):
    model.to(DEVICE)
    model.eval()

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    y_pred = []
    for X in loader:
        X = X.to(DEVICE)
        pred = model(X).detach().cpu().numpy()
        y_pred.extend(pred)

    return np.array(y_pred).flatten()