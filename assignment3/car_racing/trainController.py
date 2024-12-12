import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
from cma import CMAEvolutionStrategy
import gymnasium as gym
import logging

from torchvision import transforms
from vae import VAE
from MDNLSTM import MDNLSTM
from tqdm import tqdm


class TrainController:
    def __init__(
        self,
        controller_cls,
        vae_model: VAE,
        mdn_model: MDNLSTM,
        latent_dim,
        hidden_dim,
        action_dim,
        input_channels,
        env_name='CarRacing-v2',
        rollout_per_worker=16,
        max_steps=1000,
        device=None,
    ):
        self.controller_cls = controller_cls
        self.vae = vae_model
        self.mdn_lstm = mdn_model
        self.controller = controller_cls(latent_dim, hidden_dim)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.rollout_per_worker = rollout_per_worker
        self.max_steps = max_steps
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = gym.make(env_name)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.vae = self.vae.eval() if self.vae.cuda else self.vae.eval().to(self.device)
        self.mdn_lstm = (
            self.mdn_lstm.eval()
            if self.mdn_lstm.cuda
            else self.mdn_lstm.eval().to(self.device)
        )
        self.controller = self.controller.to(torch.float32).to(self.device)

    def rollout(self, controller, render=False):
        with torch.no_grad():
            obs, _ = self.env.reset()
            h = (
                torch.zeros(
                    1, 1, self.hidden_dim, device=self.device, dtype=torch.float32
                ),  # h
                torch.zeros(
                    1, 1, self.hidden_dim, device=self.device, dtype=torch.float32
                ),
            )  # c

            total_reward = 0
            done = False

            while not done:

                obs = self.transform(obs)
                obs_tensor = (
                    torch.from_numpy(np.array(obs, dtype=np.float32))
                    .unsqueeze(0)
                    .to(self.device)
                )
                z, _, _ = self.vae.encoder(obs_tensor)

                action = controller(z.unsqueeze(0), h[0]).cpu().detach().numpy().flatten()

                obs, reward, terminated, truncated, _ = self.env.step(action)
                if render:
                    self.env.render()

                total_reward += reward
                done = terminated or truncated

                a = torch.tensor(action, dtype=torch.float32).to(self.device)
                _, _, _, h = self.mdn_lstm(z, a.unsqueeze(0), h)

            torch.cuda.empty_cache()
            return total_reward

    def worker_process(self, param_queue, result_queue):
        controller = self.controller.to(self.device)

        while True:
            params = param_queue.get()
            if params is None:
                break

            params = torch.tensor(params, dtype=torch.float32).to(self.device)
            torch.nn.utils.vector_to_parameters(params, controller.parameters())
            rewards = [self.rollout(controller) for _ in range(self.rollout_per_worker)]
            result_queue.put(np.mean(rewards))

        torch.cuda.empty_cache()

    def train_model(self, max_iterations=100, population_size=64, num_workers=None):
        num_workers = num_workers or mp.cpu_count()
        param_queue = mp.Queue()
        result_queue = mp.Queue()

        processes = []
        for _ in range(num_workers):
            p = mp.Process(target=self.worker_process, args=(param_queue, result_queue))
            p.start()
            processes.append(p)

        controller_params = (
            torch.nn.utils.parameters_to_vector(self.controller.parameters())
            .detach()
            .cpu()
            .numpy()
        )
        cma_es = CMAEvolutionStrategy(
            controller_params, 0.5, {'popsize': population_size}
        )

        for iteration in tqdm(
            range(max_iterations), desc="Training Iterations", unit="iteration"
        ):
            solutions = cma_es.ask()

            for solution in solutions:
                param_queue.put(solution)

            rewards = [result_queue.get() for _ in range(len(solutions))]
            cma_es.tell(solutions, [-r for r in rewards])

            logging.info(
                "Iter %d/%d, Mean Reward: %.2f, Best Reward: %.2f",
                iteration + 1,
                max_iterations,
                np.mean(rewards),
                np.max(rewards),
            )

        for _ in range(num_workers):
            param_queue.put(None)
        for p in processes:
            p.join()

        torch.cuda.empty_cache()

        return self.controller
