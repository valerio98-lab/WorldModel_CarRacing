import torch 
import torch.nn as nn
import tqdm as tqdm

from MDNLSTM import MDNLSTM 
from vae import VAE
from utils import save_model, load_model


class trainMDNLSTM(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim, num_gaussians=5, resume_training=False):
        super(trainMDNLSTM, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mdn = MDNLSTM(latent_dim, action_dim, hidden_dim, num_gaussians).to(self.device)
        self.vae = VAE(input_channels=3, latent_dim=latent_dim).to(self.device)

        self.train_dataset = ...
        self.test_dataset = ...
        self.val_dataset = ...

        self.train_dataloader = ...
        self.test_dataloader = ...
        self.val_dataloader = ...

        self.optimizer = torch.optim.Adam(self.mdn.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)

        self.resume_training = resume_training
        if self.resume_training:
            self.model, self.optimizer, self.scheduler = load_model(self.model, self.optimizer, self.scheduler, 'mdn_lstm')


    def train(self, epoch): 
        self.mdn.train()
        train_loss = 0
        for idx, batch in tqdm.tqdm(enumerate(self.train_dataloader)):
            pass
