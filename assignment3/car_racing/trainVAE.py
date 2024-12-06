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

torch.manual_seed(42)


class trainVAE(nn.Module):
    def __init__(
        self,
        vae: VAE,
        dataset_path,
        input_channels=3,
        latent_dim=32,
        batch_size=32,
        epochs=10,
        episodes=20,
        episode_length=10,
        block_size=1000,
    ):
        super(trainVAE, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_channels = input_channels
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.vae = vae

        self.optimizer = optim.Adam(self.vae.parameters(), lr=1e-3)
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
            mode="frames",
            block_size=block_size,
        )
        self.dataset_val = CarRacingDataset(
            path=f"{dataset_path}/validation.pt",
            batch_size=batch_size,
            episodes=val_episodes,
            episode_length=episode_length,
            continuous=True,
            mode="frames",
            block_size=block_size,
        )
        self.dataset_test = CarRacingDataset(
            path=f"{dataset_path}/test.pt",
            batch_size=batch_size,
            episodes=test_episodes,
            episode_length=episode_length,
            continuous=True,
            mode="frames",
            block_size=block_size,
        )

        self.train_loader = DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.dataset_test, batch_size=self.batch_size, shuffle=False
        )
        self.val_loader = DataLoader(
            self.dataset_val, batch_size=self.batch_size, shuffle=False
        )

    def train(self, epoch):
        self.vae.train()
        train_loss = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            recon, mu, sigma = self.vae(batch)
            loss = self.vae.loss_function(recon, batch, mu, sigma)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        logging.info(
            "Epoch: %d, Train Loss: %f",
            epoch,
            train_loss / len(self.train_loader.dataset),
        )

        torch.cuda.empty_cache()

    def test(self):
        self.vae.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                recon, mu, sigma = self.vae(batch)
                loss = self.vae.loss_function(recon, batch, mu, sigma)
                test_loss += loss.item()

        logging.info("Test Loss: %f", test_loss / len(self.test_loader.dataset))
        torch.cuda.empty_cache()

        return test_loss / len(self.test_loader.dataset)

    def validation(self, epoch):
        self.vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                recon, mu, sigma = self.vae(batch)
                val_loss += self.vae.loss_function(recon, batch, mu, sigma)

        logging.info(
            "Epoch: %d, Validation Loss: %f",
            epoch,
            val_loss / len(self.val_loader.dataset),
        )
        torch.cuda.empty_cache()

        return val_loss / len(self.val_loader.dataset)

    def train_model(self):

        logging.info("Training model...")
        for epoch in tqdm(iterable=range(self.epochs), desc="Epochs", unit="epoch"):
            self.train(epoch)
            self.validation(epoch)
            # val_loss = self.validation(epoch)
            # self.scheduler.step(val_loss)
            # save_model(
            #     model=self.vae,
            #     optimizer=self.optimizer,
            #     epoch=epoch,
            #     model_name=self.vae_path,
            # )
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
#                     axes[0, i].imshow(batch[i].cpu().permute(1, 2, 0).squeeze(), cmap='gray')
#                     axes[0, i].axis('off')
#                     # Mostra le ricostruzioni
#                     axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).squeeze(), cmap='gray')
#                     axes[1, i].axis('off')
#                 plt.show()
#                 break

#     # Mostra le ricostruzioni usando il dataloader di test

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
# model = trainVAE(
#     dataset_path="./dataset",
#     vae_path="vae",
#     input_channels=3,
#     latent_dim=32,
#     batch_size=32,
#     epochs=2,
#     episodes=100,
#     episode_length=100,
#     block_size=10,
# )  # 300,500
# model.train_model()
# vae, _ = load_model(model=model.vae, model_name="vae")
# show_reconstructions(vae, model.train_loader, model.device)
