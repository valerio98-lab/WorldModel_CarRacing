import torch
import gymnasium as gym
import os
import numpy as np
import tqdm as tqdm


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
    def __init__(self, path, episodes=10000, episode_length=100, continuous=False, noise_type=None, mode="frames"):
        self.env = gym.make(id='CarRacing-v2', continuous=continuous)
        self.episode_length = episode_length
        self.mode = mode
        self.path = path
        self.episodes = episodes
        self.noise_type = noise_type
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        if os.path.exists(self.path):
            self.dataset = torch.load(self.path)
            print(f"Dataset loaded from {self.path}")
        else:
            directory = os.path.dirname(self.path)
            filename = os.path.basename(self.path)
            os.makedirs(directory, exist_ok=True)
            print(f"Creating dataset {filename}")
            self.dataset = self.catch_data()


    def catch_data(self):
        data = []
        lengths = []
        for episode in tqdm.tqdm(range(self.episodes)):
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

            lengths.append(len(observations))
            data.append(Episode(
                observations=torch.stack(observations),
                actions=torch.stack(actions),
                rewards=torch.stack(rewards)
            ))

        avg = int(torch.tensor(lengths, dtype=torch.float32).mean().item())
        
        dataset = []
        for elem in data:
            if len(elem.observations) >= avg:
                elem = Episode(
                    observations=elem.observations[:avg],
                    actions=elem.actions[:avg],
                    rewards=elem.rewards[:avg]
                )
            dataset.append(elem)

        torch.save(dataset, self.path)
        return dataset
    
    def brownian_noise(self, action):
        action = action + np.random.normal(0, 0.1, 3)  
        action = np.clip(action, -1, 1) 

        return action


    def __len__(self):
        if self.mode == "frames":
            return sum([len(episode.observations) for episode in self.dataset])
        elif self.mode == "episodes":
            return len(self.dataset)

    def __getitem__(self, idx):
        if self.mode == "frames":
            episode_idx = 0
            frame_idx = idx
            while frame_idx >= len(self.dataset[episode_idx].observations):
                frame_idx -= len(self.dataset[episode_idx].observations)
                episode_idx += 1
            return self.dataset[episode_idx].observations[frame_idx]
        elif self.mode == "episodes":
            return self.dataset[idx]


class LatentDataset(Dataset):
    def __init__(self, dataset_path, model_path, latent_dataset_path=None):
        self.dataset_path = dataset_path
        self.latent_dataset_path = latent_dataset_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = VAE(input_channels=3, latent_dim=32)
        self.vae, _ = load_model(self.model, load_checkpoint=False, model_name=model_path)
        self.vae.eval().to(self.device)


        if os.path.exists(self.dataset_path):
            self.dataset = torch.load(self.dataset_path)
            print(f"Len of dataset {self.dataset_path}: ", len(self.dataset))
            print(f"====>Dataset loaded from {self.dataset_path}")
        else:
            print(f"====>Dataset not found at {self.dataset_path}. Please create it first.")
            raise FileNotFoundError

        
        if os.path.exists(self.latent_dataset_path):
            self.latent_dataset = torch.load(self.latent_dataset_path)
            print(f"Len of latent dataset {self.latent_dataset_path}: ", len(self.latent_dataset))
            print(f"====>Dataset loaded from {self.latent_dataset_path}")
        else:
            filename = os.path.basename(self.latent_dataset_path)
            print(f"====>Creating dataset {filename}...")
            self.latent_dataset = self.collect_latents()


    def collect_latents(self):
        latents = []
        for episode in self.dataset:
            latent_obs = []
            with torch.no_grad():
                for idx in range(0, len(episode.observations), 32):
                    end = min(idx+32, len(episode.observations))
                    mini_batch = episode.observations[idx:end].to(self.device)
                    z, _, _ = self.vae.encoder(mini_batch)
                    latent_obs.append(z.cpu())

            latent_obs = torch.cat(latent_obs, dim=0)
            latents.append(Episode(
                observations=latent_obs,
                actions=episode.actions,
                rewards=episode.rewards
            ))

        torch.save(latents, self.latent_dataset_path)
        print(f"====>Latent dataset saved at {self.latent_dataset_path}.")



    def __len__(self):
        return len(self.latent_dataset)
    

    def __getitem__(self, idx):
        return self.latent_dataset[idx]

    



# if __name__ == "__main__":
#     # dataset = CarRacingDataset(episodes=16, episode_length=50)
#     # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
#     # for batch_observations, batch_actions, batch_rewards in dataloader:
#     #     print(batch_observations.shape, batch_actions.shape, batch_rewards.shape)

#     dataset = LatentDataset(dataset_path="dataset/train.pt", model_path="vae.pt", latent_dataset_path="dataset/latent_dataset.pt")
    
