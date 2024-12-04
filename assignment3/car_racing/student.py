import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from trainVAE import trainVAE
from trainMDNLSTM import trainMDNLSTM

from utils import test_mdn_with_visualization

from dataset import CarRacingDataset
from torch.utils.data import DataLoader



class Policy(nn.Module):
    def __init__(self, 
                    vae_path='vae.pt',
                    mdn_path='mdn.pt',
                    dataset='dataset', 
                    input_channels=3,
                    latent_dim=32,
                    hidden_dim=256,
                    num_gaussians=5,
                    batch_size_vae=32,
                    batch_size_mdn=32,
                    epochs_vae=1,
                    epochs_mdn=20,
                    episodes=10000,
                    episode_length=1000,
                    continuous=True
                ):
        
        super(Policy, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_dim = 3 if continuous else 1
        self.dataset = dataset

        self.train_vae = trainVAE(
                                vae_path=vae_path, 
                                dataset_path=dataset, 
                                input_channels=input_channels, 
                                latent_dim=latent_dim, 
                                batch_size=batch_size_vae, 
                                epochs=epochs_vae, 
                                episodes=episodes, 
                                episode_length=episode_length
                                )

        self.train_mdn = trainMDNLSTM(
                                dataset_path=dataset,
                                vae_model_path=vae_path,
                                mdn_model_path=mdn_path,
                                latent_dim=latent_dim, 
                                action_dim=self.action_dim, 
                                hidden_dim=hidden_dim, 
                                num_gaussians=num_gaussians, 
                                batch_size_vae=batch_size_vae,
                                batch_size=batch_size_mdn,
                                epochs=epochs_mdn
        )
    
    def forward(self, x):
        # TODO
        return x
    
    def act(self, state):
        # TODO
        return 

    def episode_collate_fn(self, batch):

        observations = torch.stack([episode.observations for episode in batch])
        actions = torch.stack([episode.actions for episode in batch])
        rewards = torch.stack([episode.rewards for episode in batch])
        return observations, actions, rewards


    def train(self):
        self.train_vae.train_model()
        self.train_mdn.train_model()
        # test_data = CarRacingDataset(path=f'{self.dataset}/test.pt', episodes=20, episode_length=10, continuous=True, mode='episodes')
        # loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=self.episode_collate_fn)
        # test_mdn_with_visualization(
        #     mdn=self.train_mdn.mdn,
        #     vae=self.train_mdn.vae,
        #     latent_dataloader=self.train_mdn.test_dataloader,
        #     real_dataloader=loader,
        #     device=self.device
        # )


        return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

if __name__ == "__main__":
    policy = Policy()
    policy.train()
    