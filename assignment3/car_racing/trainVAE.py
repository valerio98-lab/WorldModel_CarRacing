import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import logging
from tqdm import tqdm
import os

from vae import VAE
from dataset import CarRacingDataset, Episode
from utils import testagent, save_model

torch.manual_seed(42)


class trainVAE(nn.Module):
    def __init__(
        self,
        vae: VAE,
        dataset_path,
        input_channels=3,
        latent_dim=32,
        batch_size=32,
        epochs=1,
        episodes=20,
        episode_length=10,
        block_size=1000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(trainVAE, self).__init__()
        self.device = device
        self.input_channels = input_channels
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.vae = vae
        self.optimizer = optim.Adam(self.vae.parameters(), lr=1e-4)

        train_episodes = int(episodes * 0.8)
        val_episodes = int(episodes * 0.1)
        test_episodes = episodes - train_episodes - val_episodes

        self.dataset_train = CarRacingDataset(
            path=f"{dataset_path}/train.pt",
            batch_size=batch_size,
            episodes=train_episodes,
            episode_length=episode_length,
            continuous=True,
            block_size=block_size,
            flipping=False,
        )
        self.dataset_val = CarRacingDataset(
            path=f"{dataset_path}/validation.pt",
            batch_size=batch_size,
            episodes=val_episodes,
            episode_length=episode_length,
            continuous=True,
            block_size=block_size,
            flipping=False,
        )
        self.dataset_test = CarRacingDataset(
            path=f"{dataset_path}/test.pt",
            batch_size=batch_size,
            episodes=test_episodes,
            episode_length=episode_length,
            continuous=True,
            block_size=block_size,
            flipping=False,
        )

        self.train_loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        self.test_loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        self.val_loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        obs = [elem.observations for elem in batch]
        act = [elem.actions for elem in batch]
        rew = [elem.rewards for elem in batch]

        obs = torch.stack(obs)
        act = torch.stack(act)
        rew = torch.stack(rew)

        return obs, act, rew

    def train(self, epoch):
        self.vae.train()
        train_loss = 0
        total_frames = 0
        for obs_batch, _, _ in tqdm(self.train_loader):  # [64, 1000, 3, 64, 64]
            num_episodes, num_frames, *_ = obs_batch.shape
            perm_indices = torch.randperm(num_frames, device=obs_batch.device)
            obs_batch = obs_batch[:, perm_indices]

            for frames in range(num_frames):
                minibatch = obs_batch[:, frames].to(self.device)
                recon, mu, log_var = self.vae(minibatch)
                loss = self.vae.loss_function(
                    reconstruction=recon, x=minibatch, mu=mu, log_var=log_var
                )
                self.optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

                total_frames += minibatch.size(0)
                minibatch = minibatch.to("cpu")
                recon = recon.to("cpu")

        logging.info("Epoch: %d, Train Loss: %f", epoch, train_loss / total_frames)

        torch.cuda.empty_cache()

    def test(self):
        self.vae.eval()
        test_loss = 0
        total_frames = 0

        with torch.no_grad():
            for obs_batch, _, _ in self.test_loader:
                _, num_frames, *_ = obs_batch.shape
                for frames in range(num_frames):
                    minibatch = obs_batch[:, frames].to(self.device)

                    recon, mu, sigma = self.vae(minibatch)
                    loss = self.vae.loss_function(recon, minibatch, mu, sigma)
                    test_loss += loss.item()
                    total_frames += minibatch.size(0)
                    minibatch = minibatch.to("cpu")

        logging.info("Test Loss: %f", test_loss / total_frames)
        torch.cuda.empty_cache()

    def validation(self, epoch):
        self.vae.eval()
        val_loss = 0
        total_frames = 0

        with torch.no_grad():
            for obs_batch, _, _ in self.val_loader:
                num_episodes, num_frames, *_ = obs_batch.shape
                perm_indices = torch.randperm(num_frames, device=obs_batch.device)
                obs_batch = obs_batch[:, perm_indices]

            for frames in range(num_frames):
                minibatch = obs_batch[:, frames].to(self.device)
                recon, mu, sigma = self.vae(minibatch)
                loss = self.vae.loss_function(recon, minibatch, mu, sigma)
                val_loss += loss.item()
                total_frames += minibatch.size(0)
                minibatch = minibatch.to("cpu")

        logging.info(
            "Epoch: %d, Validation Loss: %f",
            epoch,
            val_loss / total_frames,
        )
        torch.cuda.empty_cache()

    def train_model(self, from_pretrained=False, checkpoint_path=None):

        logging.info("Training model...")
        s = 0
        if from_pretrained:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.vae.load_state_dict(checkpoint["model_state_dict"])
            s = checkpoint["epoch"]
            logging.info("Model loaded from %s", checkpoint_path)
        for epoch in tqdm(iterable=range(s, self.epochs), desc="Epochs", unit="epoch"):
            logging.info("Epoch: %d", epoch)
            self.train(epoch)
            self.validation(epoch)

            save_model(
                model=self.vae,
                optimizer=self.optimizer,
                epoch=epoch,
                model_name='vae.pt',
                checkpoint_path="vae_checkpoints",
            )
            torch.cuda.empty_cache()

        logging.info("Training completed")
        logging.info("Testing model...")
        self.test()
        torch.cuda.empty_cache()

        return self.vae
