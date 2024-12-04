import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
from cma import CMAEvolutionStrategy
import gymnasium as gym

from torchvision import transforms
from utils import load_model
from vae import VAE
from tqdm import tqdm


class TrainController:
    def __init__(
        self, controller_cls, vae_cls, mdn_lstm_cls,
        vae_path, mdn_lstm_path, latent_dim, hidden_dim, action_dim, input_channels,
        env_name='CarRacing-v2', rollout_per_worker=16, max_steps=1000, device=None
    ):
        self.controller_cls = controller_cls
        self.vae_cls = vae_cls
        self.mdn_lstm_cls = mdn_lstm_cls

        self.vae, _  = load_model(model=vae_cls(input_channels, latent_dim), model_name=vae_path, load_checkpoint=False)
        self.mdn_lstm, _ = load_model(model=mdn_lstm_cls(latent_dim, action_dim, hidden_dim), model_name=mdn_lstm_path, load_checkpoint=False)
        self.controller = controller_cls(latent_dim, hidden_dim)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.rollout_per_worker = rollout_per_worker
        self.max_steps = max_steps
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = gym.make(env_name)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        self.vae = self.vae.eval().to(self.device)
        self.mdn_lstm = self.mdn_lstm.eval().to(self.device)
        self.controller = self.controller.to(torch.float32).to(self.device)

    def rollout(self, controller, render=False):
        with torch.no_grad():
            obs, _ = self.env.reset()
            h = (torch.zeros(1, 1, self.hidden_dim, device=self.device, dtype=torch.float32),  # h
                torch.zeros(1, 1, self.hidden_dim, device=self.device, dtype=torch.float32))  # c

            total_reward = 0
            done = False

            for _ in range(self.max_steps):
                if done:
                    break

                obs = self.transform(obs)
                obs_tensor = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
                z, _, _ = self.vae.encoder(obs_tensor)
                action = controller(z, h[0]).cpu().detach().numpy().flatten()

                obs, reward, terminated, truncated, _ = self.env.step(action)
                if render:
                    self.env.render()

                total_reward += reward
                done = terminated or truncated

                a = torch.tensor(action, dtype=torch.float32).to(self.device)
                _, _, _, h = self.mdn_lstm(z, a, h)

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


    def train(self, num_iterations=100, population_size=64, num_workers=None):
        num_workers = num_workers or mp.cpu_count()
        param_queue = mp.Queue()
        result_queue = mp.Queue()

        processes = []
        for _ in range(num_workers):
            p = mp.Process(target=self.worker_process, args=(param_queue, result_queue))
            p.start()
            processes.append(p)

        controller_params = torch.nn.utils.parameters_to_vector(self.controller.parameters()).detach().cpu().numpy()
        cma_es = CMAEvolutionStrategy(controller_params, 0.5, {'popsize': population_size})

        for iteration in tqdm(range(num_iterations), desc="Training Iterations", unit="iteration"):
            solutions = cma_es.ask()
            
            for solution in solutions:
                param_queue.put(solution)

            rewards = [result_queue.get() for _ in range(len(solutions))]
            cma_es.tell(solutions, [-r for r in rewards])

            print(f"Iter {iteration + 1}/{num_iterations}, Mean Reward: {np.mean(rewards):.2f}, Best Reward: {np.max(rewards):.2f}")

        for _ in range(num_workers):
            param_queue.put(None)
        for p in processes:
            p.join()

        torch.cuda.empty_cache()

        return self.controller


# if __name__ == "__main__":
#     from controller import Controller
#     from MDNLSTM import MDNLSTM_Controller

#     mp.set_start_method('spawn')

#     latent_dim = 32
#     hidden_dim = 256
#     action_dim = 3
#     input_channels = 3
#     vae_path = './vae.pt'
#     mdn_lstm_path = './mdn_lstm.pt'


#     train_controller = TrainController(
#         controller_cls=Controller,
#         vae_cls=VAE,
#         mdn_lstm_cls=MDNLSTM_Controller,
#         vae_path=vae_path,
#         mdn_lstm_path=mdn_lstm_path,
#         latent_dim=latent_dim,
#         hidden_dim=hidden_dim,
#         action_dim=action_dim,
#         input_channels=input_channels,
#         env_name='CarRacing-v2',
#         rollout_per_worker=10,
#         max_steps=500,
#         device='cuda' if torch.cuda.is_available() else 'cpu'
#     )

#     trained_controller = train_controller.train(
#         num_iterations=10, 
#         population_size=16,  
#         num_workers=12  
# )
