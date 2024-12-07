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
from trainController import TrainController

from utils import test_mdn_with_visualization

from torch.utils.data import DataLoader

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
        batch_size_vae=32,
        batch_size_mdn=32,
        epochs_vae=1,
        epochs_mdn=20,
        episodes=10000,
        episode_length=1000,
        rollout_per_worker=16,
        max_steps=1000,
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
        )
        self.vae = train_vae.train_model()

        train_mdn = trainMDNLSTM(
            dataset_path=self.dataset_path,
            vae_model=self.vae,
            mdn_model=self.mdn,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_gaussians=self.num_gaussians,
            batch_size_vae=self.batch_size_vae,
            batch_size=self.batch_size_mdn,
            epochs=self.epochs_mdn,
            block_size=self.block_size,
            episodes=self.episodes,
        )

        self.mdn = train_mdn.train_model()

        train_controller = TrainController(
            controller_cls=Controller,
            vae_model=self.vae,
            mdn_model=self.mdn,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            input_channels=self.input_channels,
            env_name='CarRacing-v2',
            rollout_per_worker=self.rollout_per_worker,
            max_steps=self.max_steps,
            device=self.device,
        )

        self.controller = train_controller.train_model(
            num_workers=12, num_iterations=100, population_size=64
        )

        # test_data = CarRacingDataset(path=f'{self.dataset}/test.pt', episodes=20, episode_length=10, continuous=True, mode='episodes')
        # loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=self.episode_collate_fn)
        # test_mdn_with_visualization(
        #     mdn=self.train_mdn.mdn,
        #     vae=self.train_mdn.vae,
        #     latent_dataloader=self.train_mdn.test_dataloader,
        #     real_dataloader=loader,
        #     device=self.device
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
    policy = Policy()
    policy.train()
    policy.save()
    policy.load()
