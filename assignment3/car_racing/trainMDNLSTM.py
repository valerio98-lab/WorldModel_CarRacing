import torch
import torch.nn as nn
import os
from tqdm import tqdm
import logging

from torch.utils.data import DataLoader
from MDNLSTM import MDNLSTM
from vae import VAE
from dataset import LatentDataset, Episode
from utils import save_model, load_model

import matplotlib.pyplot as plt

torch.manual_seed(42)


class trainMDNLSTM(nn.Module):
    def __init__(
        self,
        dataset_path,
        vae_model: VAE,
        mdn_model: MDNLSTM,
        latent_dim,
        action_dim,
        hidden_dim,
        num_gaussians=5,
        batch_size_vae=32,
        batch_size=64,
        epochs=10,
        continuous=True,
        episodes=1000,
        block_size=1000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):

        super(trainMDNLSTM, self).__init__()
        self.device = device
        self.batch_size_vae = batch_size_vae
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.continuous = continuous

        self.mdn = mdn_model
        self.vae = vae_model
        self.vae.eval().to(self.device)

        self.train_dataset = LatentDataset(
            dataset_path=f"{self.dataset_path}/train.pt",
            vae_model=self.vae,
            batch_size=self.batch_size_vae,
            episodes=episodes * 0.8,
            block_size=block_size,
        )
        self.val_dataset = LatentDataset(
            dataset_path=f"{self.dataset_path}/validation.pt",
            vae_model=self.vae,
            batch_size=self.batch_size_vae,
            episodes=episodes * 0.1,
            block_size=block_size,
        )

        self.test_dataset = LatentDataset(
            dataset_path=f"{self.dataset_path}/test.pt",
            vae_model=self.vae,
            batch_size=self.batch_size_vae,
            episodes=episodes * 0.1,
            block_size=block_size,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        self.optimizer = torch.optim.Adam(self.mdn.parameters(), lr=1e-3)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)

    def collate_fn(self, batch):
        obs = [elem.observations for elem in batch]
        act = [elem.actions for elem in batch]
        rew = [elem.rewards for elem in batch]

        obs = torch.stack(obs)
        act = torch.stack(act)
        rew = torch.stack(rew)

        return obs, act, rew

    def train(self, epoch):
        self.mdn.train()
        train_loss = 0
        for latent_obs, act, _ in tqdm(self.train_dataloader):
            latent_obs = latent_obs.to(self.device)
            act = act.to(self.device)
            self.optimizer.zero_grad()

            # teacher forcing
            latent_obs_input = latent_obs[:, :-1, :]
            act_input = act[:, :-1, :]

            target = latent_obs[:, 1:, :]

            assert (
                latent_obs_input.shape[1] == act_input.shape[1] == target.shape[1]
            ), "Mismatch in sequence length"

            alpha, mu, sigma, _ = self.mdn(latent_obs_input, act_input)
            loss = self.mdn.mdn_loss(alpha, sigma, mu, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        print(
            f"Epoch: {epoch}, Train Loss: {train_loss / len(self.train_dataloader.dataset)}"
        )
        torch.cuda.empty_cache()

    def validation(self, epoch):
        self.mdn.eval()
        val_loss = 0
        with torch.no_grad():
            for latent_obs, act, _ in self.val_dataloader:
                latent_obs = latent_obs.to(self.device)
                act = act.to(self.device)

                latent_obs_input = latent_obs[:, :-1, :]
                act_input = act[:, :-1, :]
                target = latent_obs[:, 1:, :]

                assert (
                    latent_obs_input.shape[1] == act_input.shape[1] == target.shape[1]
                ), "Mismatch in sequence length"

                alpha, mu, sigma, _ = self.mdn(latent_obs_input, act_input)

                loss = self.mdn.mdn_loss(alpha, sigma, mu, target)
                val_loss += loss.item()

        print(
            f"Epoch: {epoch}, Validation Loss: {val_loss / len(self.val_dataloader.dataset)}"
        )

        return val_loss / len(self.val_dataloader.dataset)

    def test(self):
        self.mdn.eval()
        test_loss = 0
        with torch.no_grad():
            for latent_obs, act, _ in self.test_dataloader:

                latent_obs = latent_obs.to(self.device)
                act = act.to(self.device)

                latent_obs_input = latent_obs[:, :-1, :]
                act_input = act[:, :-1, :]
                target = latent_obs[:, 1:, :]

                assert (
                    latent_obs_input.shape[1] == act_input.shape[1] == target.shape[1]
                ), "Mismatch in sequence length"

                alpha, mu, sigma, _ = self.mdn(latent_obs_input, act_input)

                loss = self.mdn.mdn_loss(alpha, sigma, mu, target)
                test_loss += loss.item()

        print(f"Test Loss: {test_loss / len(self.test_dataloader.dataset)}")

        return test_loss / len(self.test_dataloader.dataset)

    def train_model(self):

        logging.info("Training model...")
        for epoch in tqdm(
            iterable=range(self.epochs), desc="Epochs", leave=False, unit="epoch"
        ):
            self.train(epoch)
            self.validation(epoch)
            # self.scheduler.step(val_loss)
            torch.cuda.empty_cache()

        logging.info("Training complete")
        logging.info("Testing model...")

        self.test()

        return self.mdn


# if __name__ == "__main__":
#     train = trainMDNLSTM(
#         dataset_path="dataset",
#         vae_model_path="vae.pt",
#         mdn_model_path="mdn_lstm",
#         latent_dim=32,
#         action_dim=3,
#         hidden_dim=256,
#         num_gaussians=5,
#         batch_size=32,
#         epochs=10,
#         episodes=100,
#         block_size=10,
#     )
#     train.train_model()
