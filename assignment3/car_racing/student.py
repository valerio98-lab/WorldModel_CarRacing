import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import multiprocessing as mp
import logging

from vae import VAE
from MDNLSTM import MDNLSTM
from controller import Controller
from dataset import Episode

from trainVAE import trainVAE
from trainMDNLSTM import trainMDNLSTM
from assignment3.car_racing.trainController2 import TrainController

import matplotlib.pyplot as plt


from torch.utils.data import DataLoader

from dataset import CarRacingDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore")


class Policy(nn.Module):
    def __init__(
        self,
        vae_path='vae',
        mdn_path='mdn',
        dataset_path='dataset',
        input_channels=3,
        latent_dim=32,
        hidden_dim=256,
        num_gaussians=5,
        batch_size_vae=100,
        batch_size_mdn=32,
        epochs_vae=3,
        epochs_mdn=20,
        episodes=10000,
        episode_length=1000,
        rollout_per_worker=16,
        max_steps=500,
        block_size=100,
    ):

        super(Policy, self).__init__()

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        continuous = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_dim = 3 if continuous else 1
        self.block_size = block_size
        self.dataset_path = dataset_path
        self.vae_path = vae_path
        self.mdn_path = mdn_path
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        self.batch_size_vae = batch_size_vae
        self.batch_size_mdn = batch_size_mdn
        self.epochs_vae = epochs_vae
        self.epochs_mdn = epochs_mdn
        self.episodes = episodes
        self.episode_length = episode_length
        self.rollout_per_worker = rollout_per_worker
        self.max_steps = max_steps

        self.hidden_state = (
            torch.zeros(1, 1, hidden_dim),
            torch.zeros(1, 1, hidden_dim),
        )
        self.last_action = torch.zeros(1, self.action_dim).to(self.device)

        self.vae = VAE(input_channels, latent_dim).to(self.device)

        self.mdn = MDNLSTM(
            latent_dim=latent_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            num_gaussians=num_gaussians,
        ).to(self.device)

        self.controller = Controller(latent_dim=latent_dim, hidden_dim=hidden_dim).to(
            self.device
        )

    def forward(self, state, last_action=None, hidden_state=None):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent_vector, _, _ = self.vae.encoder(state)

        _, _, _, hidden_state = self.mdn(latent_vector, last_action, hidden_state)

        with torch.no_grad():
            action = self.controller(latent_vector, hidden_state[0])

        return action, hidden_state

    def act(self, state):

        action, self.hidden_state = self.forward(
            state=state, last_action=self.last_action, hidden_state=self.hidden_state
        )

        self.last_action = action

        return action.cpu().numpy().flatten()

    def episode_collate_fn(self, batch):

        observations = torch.stack([episode.observations for episode in batch])
        actions = torch.stack([episode.actions for episode in batch])
        rewards = torch.stack([episode.rewards for episode in batch])
        return observations, actions, rewards

    def train(self):
        train_vae = trainVAE(
            vae=self.vae,
            dataset_path=self.dataset_path,
            input_channels=self.input_channels,
            latent_dim=self.latent_dim,
            batch_size=self.batch_size_vae,
            epochs=self.epochs_vae,
            episodes=self.episodes,
            episode_length=self.episode_length,
            block_size=self.block_size,
            device=self.device,
        )
        self.vae = train_vae.train_model()

    # train_mdn = trainMDNLSTM(
    #     dataset_path=self.dataset_path,
    #     vae_model=self.vae,
    #     mdn_model=self.mdn,
    #     latent_dim=self.latent_dim,
    #     action_dim=self.action_dim,
    #     hidden_dim=self.hidden_dim,
    #     num_gaussians=self.num_gaussians,
    #     batch_size_vae=self.batch_size_vae,
    #     batch_size=self.batch_size_mdn,
    #     epochs=self.epochs_mdn,
    #     block_size=self.block_size,
    #     episodes=self.episodes,
    #     device=self.device,
    # )

    # self.mdn = train_mdn.train_model()

    # train_controller = TrainController(
    #     controller_cls=Controller,
    #     vae_model=self.vae,
    #     mdn_model=self.mdn,
    #     latent_dim=self.latent_dim,
    #     hidden_dim=self.hidden_dim,
    #     action_dim=self.action_dim,
    #     input_channels=self.input_channels,
    #     env_name='CarRacing-v2',
    #     rollout_per_worker=self.rollout_per_worker,
    #     max_steps=self.max_steps,
    #     device=self.device,
    # )

    # self.controller = train_controller.train_model(
    #     num_workers=12, num_iterations=100, population_size=64
    # )

    def save(self):
        torch.save(
            {
                'vae_state_dict': self.vae.state_dict(),
                'mdn_state_dict': self.mdn.state_dict(),
                'controller_state_dict': self.controller.state_dict(),
            },
            'model.pt',
        )
        logging.info('Model saved')

    def load(self):
        checkpoint = torch.load('model.pt', map_location=self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.mdn.load_state_dict(checkpoint['mdn_state_dict'])
        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        logging.info('Models loaded from model.pt')

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    policy = Policy(
        dataset_path='dataset',
        epochs_vae=2,
    )
    policy.train()
    policy.save()
    # policy.load()
