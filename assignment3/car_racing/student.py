import warnings
import multiprocessing as mp
import logging


import gymnasium as gym
import torch
import torch.nn as nn
from torchvision import transforms


from vae import VAE
from MDNLSTM import MDNLSTM
from controller import Controller
from CMA_trainController import CMAESControllerTrainer
from CMA_trainControllerMDN import CMAESControllerTrainer_MDN
from trainVAE import trainVAE
from trainMDNLSTM import trainMDNLSTM
from dataset import Episode


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore")


class Policy(nn.Module):
    def __init__(
        self,
        vae_path='vae2',
        mdn_path='mdn2',
        dataset_path='dataset_test',
        input_channels=3,
        latent_dim=32,
        hidden_dim=256,
        num_gaussians=5,
        batch_size_vae=100,
        batch_size_mdn=1,
        epochs_vae=1,
        epochs_mdn=1,
        episodes=10,
        episode_length=50,
        rollout_per_worker=3,
        num_workers=12,
        block_size=10,
        device_controller='cpu',
        save_model_path='model.pt',
        load_model_path='model.pt',
        train_control_onlyVAE=False,
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
        self.train_control_onlyVAE = train_control_onlyVAE
        self.hidden_state = None
        self.num_generations = 1
        self.save_model_path = save_model_path
        self.load_model_path = load_model_path
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

        self.vae = VAE(input_channels, latent_dim, device=self.device).to(self.device)

        self.mdn = MDNLSTM(
            latent_dim=latent_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            num_gaussians=num_gaussians,
            device=self.device,
        ).to(self.device)

        self.controller = Controller(latent_dim=latent_dim, hidden_dim=hidden_dim).to(
            self.device_controller
        )

    def forward(self, state, hidden_state=None):
        pass

    def act(self, state):
        self.vae.eval()
        self.mdn.eval()
        self.controller.eval()

        if self.hidden_state is None:
            self.hidden_state = (
                torch.zeros(1, 1, self.hidden_dim).to(self.device),
                torch.zeros(1, 1, self.hidden_dim).to(self.device),
            )
        with torch.no_grad():
            state_tensor = self.transform(state).unsqueeze(0).to(self.device)

            z, _, _ = self.vae.encoder(state_tensor)
            z = z.to('cpu')
            h = self.hidden_state[0].squeeze(0).to('cpu')

            action = self.controller(z, h).cpu().numpy().flatten()

            a = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.hidden_state = (
                self.hidden_state[0].to(self.device),
                self.hidden_state[1].to(self.device),
            )

            _, _, _, self.hidden_state = self.mdn(z, a, self.hidden_state)

        return action

    def act_withvae(self, state):
        self.vae.eval()
        self.mdn.eval()
        self.controller.eval()

        with torch.no_grad():
            state_tensor = self.transform(state).unsqueeze(0).to(self.device)

            z, _, _ = self.vae.encoder(state_tensor)
            z = z.to('cpu')
            h = torch.zeros(1, self.hidden_dim).to('cpu')

            action = self.controller(z, h).cpu().numpy().flatten()

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
            from_pretrained=False, checkpoint_path='vae_checkpoints/checkpoint_9.pt'
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
            from_pretrained=False, checkpoint_path='mdn_checkpoints/checkpoint_24.pt'
        )
        if self.train_control_onlyVAE:
            train_controller = CMAESControllerTrainer(
                controller=self.controller,
                vae=self.vae,
                mdn=self.mdn,
                env='CarRacing-v2',
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                action_dim=self.action_dim,
                num_generations=70,
                num_workers=self.num_workers,
                device=self.device,
                rollout_per_worker=self.rollout_per_worker,
            )
        else:
            train_controller = CMAESControllerTrainer_MDN(
                controller=self.controller,
                vae=self.vae,
                env=gym.make('CarRacing-v2'),
                mdn=self.mdn,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                action_dim=self.action_dim,
                num_generations=self.num_generations,
                rollouts=1,
                device=self.device,
            )
        train_controller.train_model()

    def save(self):
        torch.save(
            {
                'vae_state_dict': self.vae.state_dict(),
                'mdn_state_dict': self.mdn.state_dict(),
                'controller_state_dict': self.controller.state_dict(),
            },
            self.save_model_path,
        )
        logging.info('Model saved')

    def load(self):
        checkpoint = torch.load(self.load_model_path, map_location=self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.mdn.load_state_dict(checkpoint['mdn_state_dict'])
        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        logging.info('Models loaded from model.pt')

        return self.vae

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
