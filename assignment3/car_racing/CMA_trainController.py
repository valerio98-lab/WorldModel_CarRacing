import torch
import cma
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
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
        """
        Classe per gestire il training di un controller usando CMA-ES.

        Parametri:
            controller: Il controller da addestrare.
            vae: Il modello VAE utilizzato per ottenere la rappresentazione latente.
            mdnrnn: Il modello MDN-RNN utilizzato (anche se qui potrebbe non essere esplicitamente sfruttato).
            env: L'environment gymnasium.
            latent_size: Dimensione del vettore latente del VAE.
            hidden_size: Dimensione dello stato nascosto utilizzato dal controller.
            action_size: Dimensione del vettore di azione.
            num_generations: Numero di generazioni CMA-ES da eseguire.
            num_workers: Numero di processi in parallelo per la valutazione.
            device: Dispositivo su cui eseguire i calcoli (es. 'cuda' o 'cpu').
        """
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

    # def _preprocess_observation(self, obs):

    #     obs_tensor = self.transform(obs).unsqueeze(0).to(self.device)
    #     return obs_tensor

    def _worker_routine(self, params):
        """
        Valuta il controller su num_episodes episodi, utilizzando i parametri forniti.
        Restituisce la ricompensa media ottenuta.
        """
        mdn = MDNLSTM(32, 3, 256, 5)
        checkpoint = torch.load('mdn_checkpoints/checkpoint_29.pt')
        mdn.load_state_dict(checkpoint['model_state_dict'])
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

    def _plot_mean_rewards(
        self, mean_rewards, output_path="./mean_reward_evolution_mdn.png"
    ):
        """
        Plotta l'evoluzione della ricompensa media per generazione.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(mean_rewards) + 1), mean_rewards, label='Mean Reward')
        plt.xlabel("Generation")
        plt.ylabel("Mean Reward")
        plt.title("Evolution of the Mean Reward Over Generations")
        plt.legend()
        plt.grid()
        plt.savefig(output_path)
        plt.close()

    def train_model(self):
        """
        Esegue il training del controller utilizzando CMA-ES:
        - Genera candidati
        - Valuta in parallelo
        - Aggiorna i parametri CMA-ES
        - Verifica early stopping
        - Plotta i risultati
        """
        print("Training the controller using CMA-ES...")

        num_params = sum(
            p.numel() for p in self.controller.parameters() if p.requires_grad
        )

        # Inizializza CMA-ES
        es = cma.CMAEvolutionStrategy(num_params * [0], 0.5)

        mean_rewards = []

        # Pool di processi per la valutazione parallela
        with mp.Pool(processes=self.num_workers) as pool:
            evaluate_partial = partial(self._worker_routine)

            for generation in range(self.num_generations):
                # Genera soluzioni candidato
                solutions = es.ask()

                # Valuta le soluzioni in parallelo
                rewards = list(
                    tqdm(
                        pool.imap(evaluate_partial, solutions),
                        total=len(solutions),
                        desc=f"Generation {generation+1}/{self.num_generations}",
                    )
                )

                # CMA-ES minimizza, invertiamo il segno delle ricompense
                neg_rewards = [-r for r in rewards]
                es.tell(solutions, neg_rewards)
                es.logger.add()
                es.disp()

                # Calcolo della media e della migliore ricompensa
                mean_reward = -sum(neg_rewards) / len(neg_rewards)
                mean_rewards.append(mean_reward)
                best_reward = -min(neg_rewards)

                print(
                    f"Generation {generation + 1}/{self.num_generations}, "
                    f"Mean Reward: {mean_reward:.4f}, Best Reward: {best_reward:.4f}"
                )

                # Early stopping se non migliora
                if generation > 5 and abs(mean_rewards[-1] - mean_rewards[-5]) < 1e-3:
                    print("Early stopping triggered.")
                    break

        torch.cuda.empty_cache()
        best_params = es.result.xbest
        self.controller.set_params(best_params)

        # Plot dei risultati
        self._plot_mean_rewards(mean_rewards, output_path="./mean_reward_evolution.png")

        return self.controller, best_params
