import torch
import gymnasium as gym
import os
import numpy as np
from tqdm import tqdm
import math
import glob
import logging
import re
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass
from utils import load_model, search_files
from vae import VAE

torch.manual_seed(42)


@dataclass
class Episode:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor


class CarRacingDataset(Dataset):
    def __init__(
        self,
        path,
        batch_size=32,
        episodes=10000,
        episode_length=1000,
        continuous=False,
        block_size=1000,
        flipping=True,
    ):
        self.env = gym.make(id="CarRacing-v2", continuous=continuous)
        self.horizontal_flip = transforms.RandomHorizontalFlip() if flipping else None
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.path = path
        self.block_size = block_size
        self.episodes = episodes
        self.transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((64, 64)), transforms.ToTensor()]
        )

        if not self.path.endswith(".pt"):
            logging.error("Dataset path must be a .pt file.")
            raise ValueError("Dataset path must be a .pt file.")

        self.avg = 0
        self.path = Path(path.split(".pt")[0])
        self.lengths = []

        n_files = math.ceil(episodes / self.block_size)
        self.current_block = 0

        self.dataset_files = search_files(self.path)

        if len(self.dataset_files) == n_files:
            logging.info(
                "Dataset %s already exists. Found %d files.",
                self.path,
                len(self.dataset_files),
            )
            self.dataset = torch.load(self.dataset_files[0])

        else:
            directory = os.path.dirname(self.path)
            filename = os.path.basename(self.path)
            os.makedirs(directory, exist_ok=True)
            logging.info(
                "Not enough files for the dataset or does not exists %s. Creating dataset %s",
                self.path,
                filename,
            )
            self.catch_data()
            self.dataset_files = search_files(self.path)
            self.dataset = torch.load(self.dataset_files[0])

    def catch_data(self):
        data = []
        lengths = []
        for episodes in tqdm(range(self.episodes)):
            observations, actions, rewards = [], [], []
            obs, _ = self.env.reset()
            for _ in range(self.episode_length):
                action = self.env.action_space.sample()

                next_obs, reward, done, _, _ = self.env.step(action)
                obs_tensor = self.transform(obs)
                observations.append(obs_tensor)
                actions.append(torch.tensor(action, dtype=torch.float32))
                rewards.append(torch.tensor(reward, dtype=torch.float32))
                obs = next_obs

                if done:
                    break
            lengths.append(len(observations))
            data.append(
                Episode(
                    observations=torch.stack(observations),
                    actions=torch.stack(actions),
                    rewards=torch.stack(rewards),
                )
            )
            if len(data) % self.block_size == 0:
                torch.save(data, f"{self.path}_{(episodes+1)}.pt")
                data = []
                torch.cuda.empty_cache()

        if len(data) > 0:
            directory = os.path.dirname(self.path)
            filename = os.path.basename(self.path)
            torch.save(data, os.path.join(directory, f"{filename}_{len(data)}.pt"))
            data = []
            torch.cuda.empty_cache()

        avg = int(torch.tensor(lengths, dtype=torch.float32).mean().item())
        self.cut_data(avg)

    def cut_data(self, avg):
        logging.info("Levelling dataset...")
        path = f"{self.path}_*.pt"
        for dire, _, files in os.walk(path):
            for file in files:
                data = torch.load(file)
                dataset = []
                for episode in data:
                    if len(episode.observations) > avg:
                        episode = Episode(
                            observations=episode.observations[:avg],
                            actions=episode.actions[:avg],
                            rewards=episode.rewards[:avg],
                        )
                    dataset.append(episode)
                torch.save(dataset, os.path.join(dire, file))
        logging.info("Dataset levelled.")

    def _load_block(self, block_idx):
        if block_idx < len(self.dataset_files):
            self.dataset = torch.load(self.dataset_files[block_idx])
            self.current_block = block_idx
        else:
            raise IndexError(f"Block {block_idx} does not exists.")

    def __len__(self):
        return len(self.dataset_files) * len(self.dataset)

    def __getitem__(self, idx):
        block_idx = math.ceil(idx / self.block_size) - 1
        if block_idx != self.current_block:
            self._load_block(block_idx)
        episode_idx = idx % len(self.dataset)
        episode = self.dataset[episode_idx]
        if self.horizontal_flip is not None:
            obs_flipped = [self.horizontal_flip(obs) for obs in episode.observations]
            episode = Episode(
                observations=torch.stack(obs_flipped),
                actions=episode.actions,
                rewards=episode.rewards,
            )
        return episode


class LatentDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        vae_model: VAE,
        batch_size=32,
        episodes=10000,
        block_size=1000,
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.block_size = block_size
        self.vae = vae_model
        self.vae.eval().to(self.device)

        self.current_block = 0
        self.dataset = []

        if not self.dataset_path.endswith(".pt"):
            logging.error("Dataset path must be a .pt file.")
            raise ValueError("Dataset path must be a .pt file.")

        n_files = math.ceil(episodes / self.block_size)

        self.path = dataset_path.split(".pt")[0]
        self.dataset_files = search_files(self.path)
        self.latent_dataset_files = search_files(f"{self.path}-latent")
        self.current_block = 0

        if len(self.dataset_files) >= n_files:
            logging.info(
                "Dataset %s exists. Found %d files.",
                self.dataset_path,
                len(self.dataset_files),
            )
            self.dataset = torch.load(self.dataset_files[0])
        else:
            logging.error("Dataset %s not found. Create it first.", self.path)
            raise FileNotFoundError(f"Dataset {self.path} not found.")

        if len(self.latent_dataset_files) == n_files:
            logging.info(
                "Latent dataset files found: %d.", len(self.latent_dataset_files)
            )
            self.latent_dataset = torch.load(self.latent_dataset_files[0])
            self.dataset = []
        else:
            logging.info("Latent dataset not found. Creating latent dataset...")
            self.collect_latents()
            self.latent_dataset_files = search_files(f"{self.path}-latent")
            self.latent_dataset = torch.load(self.latent_dataset_files[0])

    def collect_latents(self):
        latents = []
        for idx_f, files in enumerate(self.dataset_files):
            self.dataset = torch.load(files)

            for idx_ep, episode in enumerate(self.dataset):
                latent_obs = []
                with torch.no_grad():
                    for idx in range(0, len(episode.observations), self.batch_size):
                        end = min(idx + self.batch_size, len(episode.observations))
                        mini_batch = episode.observations[idx:end].to(self.device)
                        z, _, _ = self.vae.encoder(mini_batch)
                        latent_obs.append(z.cpu())
                latent_obs = torch.cat(latent_obs, dim=0)

                latents.append(
                    Episode(
                        observations=latent_obs,
                        actions=episode.actions,
                        rewards=episode.rewards,
                    )
                )
                print(self.block_size)
                if len(latents) % self.block_size == 0:
                    torch.save(
                        latents,
                        f"{self.path}-latent_{((idx_f+1) * (idx_ep+1))}.pt",
                    )
                    latents = []
                    torch.cuda.empty_cache()

            if len(latents) > 0:
                torch.save(latents, f"{self.path}-latent_{len(latents)}.pt")
                latents = []
                torch.cuda.empty_cache()

        logging.info("Latent dataset saved at %s.", self.path)

    def _load_block(self, block_idx):
        if block_idx < len(self.latent_dataset_files):
            self.latent_dataset = torch.load(self.latent_dataset_files[block_idx])
            self.current_block = block_idx
            # logging.info(
            #     "Loaded block %d with %d episodes.", block_idx, len(self.latent_dataset)
            # )
        else:
            raise IndexError(
                f"Block index {block_idx} out of range. Dataset files: {self.latent_dataset_files}"
            )

    def __len__(self):
        return len(self.latent_dataset_files) * len(self.latent_dataset)

    def __getitem__(self, idx):
        block_idx = math.ceil(idx / self.block_size) - 1
        if block_idx != self.current_block:
            self._load_block(block_idx)
        episode_idx = idx % len(self.latent_dataset)
        return self.latent_dataset[episode_idx]
