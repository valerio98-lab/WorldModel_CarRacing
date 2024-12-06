import torch
import gymnasium as gym
import os
import numpy as np
from tqdm import tqdm
import math
import glob 
import logging


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass
from utils import load_model
from vae import VAE

torch.manual_seed(42)



@dataclass
class Episode:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor


class CarRacingDataset(Dataset):
    def __init__(self, path, batch_size=32, episodes=10000, episode_length=1000, continuous=False, noise_type=None, mode="frames"):
        self.env = gym.make(id='CarRacing-v2', continuous=continuous)
        self.episode_length = episode_length
        self.mode = mode
        self.batch_size = batch_size
        self.path = path
        self.episodes = episodes
        self.noise_type = noise_type
        self.transform = transforms.Compose([  
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),      
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.avg = 0
        self.path = path.split(".")[0]
        self.lengths = []
        n_files = math.ceil(episodes/10)
        
        self.dataset_files = sorted([f for f in glob.glob(f"{self.path}_*.pt")])
        self.current_block = 0

        if len(self.dataset_files) == n_files:
            logging.info("Dataset %s already exists. Found %d files.", self.path, len(self.dataset_files))
            self.dataset = torch.load(self.dataset_files[0])
        
        else:
            directory = os.path.dirname(self.path)
            filename = os.path.basename(self.path)
            os.makedirs(directory, exist_ok=True)
            logging.info("Not enough files for the dataset or does not exists %s. Creating dataset %s", self.path, filename)
            self.catch_data()
            self.dataset_files = sorted([f for f in glob.glob(f"{self.path}_*.pt")])
            self.dataset = torch.load(self.dataset_files[0])


    def catch_data(self):
        data = []
        for episodes in tqdm(range(self.episodes)):
            observations, actions, rewards = [], [], []
            obs, _ = self.env.reset()
            for _ in range(self.episode_length):
                action = self.env.action_space.sample()  #noise_type='white'
                if self.noise_type == 'brownian':
                    action = self.brownian_noise(action)
                

                next_obs, reward, done, _, _ = self.env.step(action)
                obs_tensor = self.transform(obs)
                observations.append(obs_tensor)
                actions.append(torch.tensor(action, dtype=torch.float32))
                rewards.append(torch.tensor(reward, dtype=torch.float32))
                obs = next_obs

                if done:
                    break
            self.lengths.append(len(observations))                   
            data.append(Episode(
                observations=torch.stack(observations),
                actions=torch.stack(actions),
                rewards=torch.stack(rewards)
            ))
            if len(data) % 10 == 0:
                torch.save(data, f"{self.path}_{episodes+1}.pt")
                del data[:]
                torch.cuda.empty_cache()
        
        if len(data) > 0:
            torch.save(data, f"{self.path}_{len(data)}.pt")
            del data[:]
            data = []
            torch.cuda.empty_cache()
        

        self.avg = int(torch.tensor(self.lengths, dtype=torch.float32).mean().item())
        self.cut_data()

    def cut_data(self):
        logging.info("Levelling dataset...")
        path = f"{self.path}_*.pt"
        for dire, _, files in os.walk(path):
            for file in files:
                data = torch.load(file)
                dataset = []
                for episode in data:
                    if len(episode.observations) > self.avg:
                        episode = Episode(
                            observations=episode.observations[:self.avg],
                            actions=episode.actions[:self.avg],
                            rewards=episode.rewards[:self.avg]
                        )
                    dataset.append(episode)
                torch.save(dataset, os.path.join(dire, file))
        logging.info("Dataset levelled.")


    
    def brownian_noise(self, action):
        action = action + np.random.normal(0, 0.1, 3)  
        action = np.clip(action, -1, 1) 

        return action


    def _load_block(self, block_idx):
        if block_idx < len(self.dataset_files):
            self.dataset = torch.load(self.dataset_files[block_idx])
            self.current_block = block_idx
            print(f"Uploaded block {block_idx + 1}/{len(self.dataset_files)}: {self.dataset_files[block_idx]}")
        else:
            raise IndexError(f"Block {block_idx} does not exists.")



    def __len__(self):
        if self.mode == "frames":
            return sum(len(episode.observations) for episode in self.dataset) + \
                   (len(self.dataset_files) - self.current_block - 1) * len(self.dataset[0].observations)
        elif self.mode == "episodes":
            return len(self.dataset_files) * len(self.dataset)




    def __getitem__(self, idx):
        if self.mode == "frames":
            frame_count = 0
            for block_idx in range(len(self.dataset_files)):
                self._load_block(block_idx)
                for episode in self.dataset:
                    if frame_count + len(episode.observations) > idx:
                        return episode.observations[idx - frame_count]
                    frame_count += len(episode.observations)
            raise IndexError(f"Index {idx} out of range.")

        elif self.mode == "episodes":
            block_idx = math.ceil(idx/10)
            if block_idx != self.current_block:
                self._load_block(block_idx)
            episode_idx = idx % len(self.dataset)
            return self.dataset[episode_idx]


class LatentDataset(Dataset):
    def __init__(self, dataset_path, model_path, batch_size=32, latent_dataset_path=None, episodes=10000):
        self.dataset_path = dataset_path
        self.latent_dataset_path = latent_dataset_path.split(".")[0]    
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = VAE(input_channels=3, latent_dim=32)
        self.vae, _ = load_model(self.model, load_checkpoint=False, model_name=model_path)
        self.vae.eval().to(self.device)


        self.current_block = 0
        self.dataset = []

        n_files = math.ceil(episodes/10)
        path = dataset_path.split(".")[0]
        self.dataset_files = sorted([f for f in glob.glob(f"{path}_*.pt")])
        self.latent_dataset_files = sorted([f for f in glob.glob(f"{self.latent_dataset_path}_latent_*.pt")])
        self.current_block = 0

        if len(self.dataset_files) == n_files:
            logging.info("Dataset %s exists. Found %d files.", self.dataset_path, len(self.dataset_files))
            self.dataset = torch.load(self.dataset_files[0])
        else:
            logging.error("Dataset %s not found. Create it first.", self.dataset_path)
            raise FileNotFoundError(f"Dataset {self.dataset_path} not found.")
        
        if len(self.latent_dataset_files) == n_files:
            logging.info("Latent dataset files found: %d.", len(self.latent_dataset_files))
            self.latent_dataset = torch.load(self.latent_dataset_files[0])
            del self.dataset[:]
        else:
            logging.info("Latent dataset not found. Creating latent dataset...")
            self.collect_latents()
            self.latent_dataset_files = sorted([f for f in glob.glob(f"{self.latent_dataset_path}_latent_*.pt")])
            self.latent_dataset = torch.load(self.latent_dataset_files[0])



    def collect_latents(self):
            latents = []
            for idx_f, files in enumerate(self.dataset_files):
                del self.dataset[:]
                self.dataset = torch.load(files)

                for idx_ep, episode in enumerate(self.dataset):
                    latent_obs = []
                    with torch.no_grad():
                        for idx in range(0, len(episode.observations), self.batch_size):
                            end = min(idx+self.batch_size, len(episode.observations))
                            mini_batch = episode.observations[idx:end].to(self.device)
                            print(f"Mini-batch shape: {mini_batch.shape}")
                            z, _, _ = self.vae.encoder(mini_batch)
                            print(f"Latent shape: {z.shape}")
                            latent_obs.append(z.cpu())

                    latent_obs = torch.cat(latent_obs, dim=0)
                    #print(f"Latent observations shape for episode: {latent_obs.shape}")

                    latents.append(Episode(
                        observations=latent_obs,
                        actions=episode.actions,
                        rewards=episode.rewards
                    ))
                    #print(f"Number of episodes in latent dataset: {len(latents)}")

                    if len(latents) % 10 == 0:
                        torch.save(latents, f"{self.latent_dataset_path}_latent_{((idx_f+1) * (idx_ep+1))}.pt")
                        del latents[:]
                        latents = []
                        torch.cuda.empty_cache()

                if len(latents) > 0:
                    torch.save(latents, f"{self.latent_dataset_path}_{len(latents)}.pt")
                    del latents[:]
                    torch.cuda.empty_cache()

            logging.info("Latent dataset saved at %s.", self.latent_dataset_path)


    def _load_block(self, block_idx):
        if block_idx < len(self.latent_dataset_files):
            self.latent_dataset = torch.load(self.latent_dataset_files[block_idx])
            self.current_block = block_idx
            logging.info("Loaded block %d with %d episodes.", block_idx, len(self.latent_dataset))
        else:
            raise IndexError(f"Block index {block_idx} out of range. Dataset files: {self.latent_dataset_files}")


    def __len__(self):
        return len(self.latent_dataset_files) * len(self.latent_dataset)


    def __getitem__(self, idx):
        block_idx = math.ceil(idx/10)
        if block_idx != self.current_block:
            self._load_block(block_idx)
        episode_idx = idx % len(self.latent_dataset_files)
        return self.latent_dataset[episode_idx]



if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    vae = VAE(input_channels=3, latent_dim=32)
    torch.save(vae.state_dict(), "vae.pt")

    data = CarRacingDataset(path='dataset/train.pt', episodes=40, episode_length=100, mode="episodes")

    dataset = LatentDataset(dataset_path='dataset/train.pt', model_path='vae.pt' ,episodes=40, latent_dataset_path='dataset/latent_train.pt')
    print(len(dataset))
    print(dataset[0].observations.shape)
    print(dataset[0].actions.shape)
    print(dataset[0].rewards.shape)


# if __name__ == "__main__":
#     dataset = CarRacingDataset(path="giggi/dataset.pt", episodes=10, episode_length=100)
#     print(len(dataset))
#     print(dataset[0].shape)
#     latent_dataset = LatentDataset(dataset_path="giggi/dataset.pt", model_path="vae.pt", batch_size=32, latent_dataset_path="latent_dataset.pt")
#     print(len(latent_dataset))
#     print(latent_dataset[0].shape)
#     print(latent_dataset[0].actions.shape)
#     print(latent_dataset[0].rewards.shape)
#     print(latent_dataset[0].observations.shape)
#     print(latent_dataset[0].observations[0].shape)
#     print(latent_dataset[0].observations[0].dtype)
#     print(latent_dataset[0].observations[0].device)
#     print(latent_dataset[0].observations[0].mean())
#     print(latent_dataset[0].observations[0].std())
#     print(latent_dataset[0].observations[0].min())
#     print(latent_dataset[0].observations[0].max())

