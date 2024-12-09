import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
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
        warmup_steps = 1000
        # self.scheduler = LambdaLR(
        #     self.optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps)
        # )
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)

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

                # self.scheduler.step()

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

    def train_model(self):

        logging.info("Training model...")
        for epoch in tqdm(iterable=range(self.epochs), desc="Epochs", unit="epoch"):
            logging.info("Epoch: %d", epoch)
            self.train(epoch)
            self.validation(epoch)
            # if epoch in (0, 1, 2, 3, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19):
            #     ep = self.dataset_test[0]
            #     testagent(ep, self.vae, self.device)
            # val_loss = self.validation(epoch)
            # self.scheduler.step(val_loss)
            save_model(
                model=self.vae,
                optimizer=self.optimizer,
                epoch=epoch,
                model_name='vae.pt',
            )
            torch.cuda.empty_cache()

        logging.info("Training completed")
        logging.info("Testing model...")
        self.test()
        torch.cuda.empty_cache()

        return self.vae


# if __name__ == "__main__":
#     import logging

#     import matplotlib.pyplot as plt

#     def show_reconstructions(model, dataloader, device):
#         model.eval()
#         with torch.no_grad():
#             for batch in dataloader:
#                 batch = batch.to(device)
#                 recon, _, _ = model(batch)
#                 # Mostra le immagini originali e quelle ricostruite
#                 fig, axes = plt.subplots(2, 10, figsize=(15, 3))
#                 for i in range(10):
#                     # Mostra le immagini originali
#                     axes[0, i].imshow(
#                         batch[i].cpu().permute(1, 2, 0).squeeze(), cmap='gray'
#                     )
#                     axes[0, i].axis('off')
#                     # Mostra le ricostruzioni
#                     axes[1, i].imshow(
#                         recon[i].cpu().permute(1, 2, 0).squeeze(), cmap='gray'
#                     )
#                     axes[1, i].axis('off')
#                 plt.show()
#                 break

#     # Mostra le ricostruzioni usando il dataloader di test

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
# model = trainVAE(
#     dataset_path="./dataset_test",
#     vae=VAE(input_channels=3, latent_dim=32).to,
#     input_channels=3,
#     latent_dim=32,
#     batch_size=32,
#     epochs=2,
#     episodes=300,
#     episode_length=500,
#     block_size=100,
# )  # 300,500
# model.train_model()
# vae = model.vae
# show_reconstructions(vae, model.test_loader, model.device)
