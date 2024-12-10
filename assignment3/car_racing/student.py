import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import multiprocessing as mp
import logging

from torchvision import transforms
from vae import VAE
from MDNLSTM import MDNLSTM
from controller import Controller
from dataset import Episode

from trainVAE import trainVAE
from trainMDNLSTM import trainMDNLSTM
from trainController import TrainController

import matplotlib.pyplot as plt


from torch.utils.data import DataLoader

from dataset import CarRacingDataset
from utils import testagent, save_model

# from Controller_trainer_tester3 import CMAESControllerTrainer
from CMA_trainController import CMAESControllerTrainer
from newtest import CMAESControllerTrainer_MDN

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore")


class Policy(nn.Module):
    def __init__(
        self,
        vae_path='vae',
        mdn_path='mdn',
        dataset_path='dataset_2',
        input_channels=3,
        latent_dim=32,
        hidden_dim=256,
        num_gaussians=5,
        batch_size_vae=100,
        batch_size_mdn=32,
        epochs_vae=19,
        epochs_mdn=80,
        episodes=300,
        episode_length=500,
        rollout_per_worker=3,
        num_workers=8,
        block_size=10,
        device_controller='cpu',
    ):

        super(Policy, self).__init__()

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.continuous = True
        self.device_controller = device_controller
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_dim = 3 if self.continuous else 1
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
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )

        self.hidden_state = (
            torch.zeros(1, 1, hidden_dim).to(self.device),
            torch.zeros(1, 1, hidden_dim).to(self.device),
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
            self.device_controller
        )

    def forward(self, state, last_action=None, hidden_state=None):
        pass

    def act(self, state):
        self.vae.eval()
        self.mdn.eval()
        self.controller.eval()
        with torch.no_grad():
            state_tensor = self.transform(state).unsqueeze(0).to(self.device)

            z, _, _ = self.vae.encoder(state_tensor)
            z = z.to('cpu')
            h = torch.zeros(1, self.hidden_dim).to('cpu')

            # self.controller.to('cpu')
            # print(z.shape, h.shape)
            # exit()
            action = self.controller(z, h).cpu().numpy().flatten()
            print(action)
        return action

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
        self.vae = train_vae.train_model(
            from_pretrained=True, checkpoint_path='vae_checkpoints/checkpoint_19.pt'
        )

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
            device=self.device,
        )

        self.mdn = train_mdn.train_model(
            from_pretrained=False, checkpoint_path='mdn_checkpoints/checkpoint_29.pt'
        )

        # train_controller = CMAESControllerTrainer(
        #     controller=self.controller,
        #     vae=self.vae,
        #     # mdn_checkpoint_path='mdn_checkpoints/checkpoint_29.pt',
        #     mdn=self.mdn,
        #     env='CarRacing-v2',
        #     latent_dim=self.latent_dim,
        #     hidden_dim=self.hidden_dim,
        #     action_dim=self.action_dim,
        #     num_generations=50,
        #     num_workers=self.num_workers,
        #     device=self.device,
        #     rollout_per_worker=self.rollout_per_worker,
        # )
        # train_controller.train()

        train_controller = CMAESControllerTrainer_MDN(
            controller=self.controller,
            vae=self.vae,
            env=gym.make('CarRacing-v2'),
            mdn=self.mdn,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            num_generations=100,
            rollouts=3,
        )
        train_controller.train_model()

        # self.controller, best_params = train_controller_cmaes(
        #     controller=self.controller,
        #     vae=self.vae,
        #     mdnrnn=self.mdn,
        #     env=gym.make('CarRacing-v2'),
        #     latent_size=self.latent_dim,
        #     hidden_size=self.hidden_dim,
        #     action_size=self.action_dim,
        #     num_generations=50,
        # )

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
        #     device=self.device,
        # )

        # self.controller = train_controller.train_model(
        #     num_workers=12, max_iterations=10, population_size=8
        # )

    def save(self):
        torch.save(
            {
                'vae_state_dict': self.vae.state_dict(),
                'mdn_state_dict': self.mdn.state_dict(),
                'controller_state_dict': self.controller.state_dict(),
            },
            'model_mdn.pt',
        )
        logging.info('Model saved')

    def load(self):
        checkpoint = torch.load('model_mdn.pt', map_location=self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.mdn.load_state_dict(checkpoint['mdn_state_dict'])
        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        logging.info('Models loaded from model.pt')

        return self.vae

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret


if __name__ == "__main__":
    torch.cuda.empty_cache()
    mp.set_start_method('spawn', force=True)
    policy = Policy(
        dataset_path='dataset_2',
        epochs_vae=19,
        epochs_mdn=30,
        episodes=300,
        episode_length=500,
        block_size=10,
        batch_size_vae=64,
        batch_size_mdn=10,
        rollout_per_worker=8,
    )
    policy.train()
    policy.save()
    policy.load()
    # vae = VAE(3, 32).to('cuda')
    # mdn = MDNLSTM(32, 3, 256, 5).to('cuda')
    # checkpoint = torch.load('vae_checkpoints/checkpoint_19.pt')
    # checkpoint_mdn = torch.load('mdn_checkpoints/checkpoint_29.pt')
    # vae.load_state_dict(checkpoint['model_state_dict'])
    # mdn.load_state_dict(checkpoint_mdn['model_state_dict'])

    # dataset = CarRacingDataset(
    #     'dataset_test3/test.pt',
    #     block_size=50,
    #     episodes=50,
    #     episode_length=300,
    #     continuous=True,
    # )
    # for i in range(10):
    #     testagent(dataset[i], vae=vae, mdn=mdn)
