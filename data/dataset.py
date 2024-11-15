import torch
from torch.utils.data import Dataset

class ConditionalDataset(Dataset):
    """
    A Dataset object holding a tuple (x,y): observed and auxiliary variable
    """

    def __init__(self, X, Y, device='cpu', S=None, latent_dim=None):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(Y).to(device)  # if discrete, then this should be one_hot
        self.s = torch.from_numpy(S).to(device) if S is not None else None
        self.len = self.x.shape[0]
        self.aux_dim = self.y.shape[1]
        self.data_dim = self.x.shape[1]
        if latent_dim is not None:
            self.latent_dim = latent_dim
        else:
            self.latent_dim = self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.s is not None:
            return self.x[index], self.y[index], self.s[index]
        else:
            return self.x[index], self.y[index]

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim