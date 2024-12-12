import multiprocessing as mp
from functools import partial

import torch
import cma
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import torchvision.transforms as transforms

from MDNLSTM import MDNLSTM


class CMAESControllerTrainer:
    def __init__(
        self,
        controller,
        vae,
        mdn,
        env,
        latent_dim,
        hidden_dim,
        action_dim,
        num_generations,
        num_workers=4,
        rollout_per_worker=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):

        self.controller = controller
        self.vae = vae
        self.mdn = mdn
        self.env = env
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_generations = num_generations
        self.num_workers = num_workers
        self.rollout_per_worker = rollout_per_worker
        self.device = device

        # Preprocessing delle osservazioni
        self.transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((64, 64)), transforms.ToTensor()]
        )

    def worker_routine(self, params):
        self.controller.set_controller_parameters(params)
        total_reward = 0
        env = gym.make("CarRacing-v2")

        for _ in range(self.rollout_per_worker):
            obs, _ = env.reset()
            h = torch.zeros(1, self.hidden_dim)
            done = False
            truncated = False
            episode_reward = 0

            while not (done or truncated):
                with torch.no_grad():
                    obs_tensor = self.transform(obs).unsqueeze(0).to(self.device)
                    z, _, _ = self.vae.encoder(obs_tensor)
                    z = z.to('cpu')
                    h = h.to('cpu')

                    action = self.controller(z, h).detach().numpy().flatten()
                    z = z.unsqueeze(0).to(self.device)
                    obs, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward

            total_reward += episode_reward

        torch.cuda.empty_cache()
        return total_reward / self.rollout_per_worker

    def plot_mean_rewards(self, means, output_path="./mean_reward_evolution_mdn.png"):

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(means) + 1), means, label='Mean Reward')
        plt.xlabel("Generation")
        plt.ylabel("Mean Reward")
        plt.legend()
        plt.grid()
        plt.savefig(output_path)
        plt.close()

    def train_model(self):

        print("Training the controller using CMA-ES...")

        num_params = sum(
            p.numel() for p in self.controller.parameters() if p.requires_grad
        )

        es = cma.CMAEvolutionStrategy(num_params * [0], 0.5)

        means = []

        with mp.Pool(processes=self.num_workers) as pool:
            evaluate_partial = partial(self.worker_routine)

            for generation in range(self.num_generations):
                solutions = es.ask()

                rewards = list(
                    tqdm(
                        pool.imap(evaluate_partial, solutions),
                        total=len(solutions),
                        desc=f"Generation {generation+1}/{self.num_generations}",
                    )
                )

                neg_rewards = [-r for r in rewards]
                es.tell(solutions, neg_rewards)
                es.logger.add()
                es.disp()

                mean_reward = -sum(neg_rewards) / len(neg_rewards)
                means.append(mean_reward)
                best_reward = -min(neg_rewards)

                print(
                    f"Generation {generation + 1}/{self.num_generations}, "
                    f"Mean Reward: {mean_reward:.4f}, Best Reward: {best_reward:.4f}"
                )

        torch.cuda.empty_cache()
        best_params = es.result.xbest
        self.controller.set_controller_parameters(best_params)

        self.plot_mean_rewards(means, output_path="./mean_reward_evolution.png")

        return self.controller, best_params
