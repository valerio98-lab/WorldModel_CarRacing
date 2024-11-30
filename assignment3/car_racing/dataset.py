import torch
import gymnasium as gym
import os
import numpy as np
import tqdm as tqdm


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass



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



if __name__ == "__main__":
    dataset = CarRacingDataset(episodes=16, episode_length=50)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for batch_observations, batch_actions, batch_rewards in dataloader:
        print(batch_observations.shape, batch_actions.shape, batch_rewards.shape)
