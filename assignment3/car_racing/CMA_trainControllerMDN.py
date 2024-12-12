import torch
from torchvision import transforms
import cma
import tqdm
import logging
import matplotlib.pyplot as plt


class CMAESControllerTrainer_MDN:
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
        rollouts=1,
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
        self.device = device
        self.rollouts = rollouts
        self.env = env

        self.num_params = sum(
            p.numel() for p in self.controller.parameters() if p.requires_grad
        )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )

        self.vae = self.vae.to(self.device)
        self.mdn = self.mdn.to(self.device)
        self.controller = self.controller.to(self.device)

    def rollout(self, params):
        self.controller.set_controller_parameters(params)
        total_reward = 0

        for _ in range(self.rollouts):
            obs, _ = self.env.reset()
            h = (
                torch.zeros(1, 1, self.hidden_dim, device='cpu'),
                torch.zeros(1, 1, self.hidden_dim, device='cpu'),
            )
            done = False
            truncated = False
            episode_reward = 0

            while not (done or truncated):
                with torch.no_grad():
                    obs_tensor = self.transform(obs).unsqueeze(0).to(self.device)

                    z, _, _ = self.vae.encoder(obs_tensor)
                    z = z.to('cpu')
                    h_0 = h[0].squeeze(0).to('cpu')

                    action = self.controller(z, h_0).detach().numpy().flatten()
                    obs, reward, done, truncated, _ = self.env.step(action)

                    episode_reward += reward
                    a = (
                        torch.tensor(action, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    h = (h[0].to(self.device), h[1].to(self.device))
                    _, _, _, h = self.mdn(z, a, h)

            total_reward += episode_reward

        torch.cuda.empty_cache()
        return total_reward / self.rollouts

    def train_model(self):
        """
        Train the controller using CMA-ES and plot the evolution of the mean reward over generations.
        """

        logging.info("Training the controller using CMA-ES...")

        es = cma.CMAEvolutionStrategy(self.num_params * [0], 0.5)

        mean_rewards = []

        for generation in tqdm.tqdm(
            range(self.num_generations),
            desc="Training controller",
            total=self.num_generations,
            leave=False,
        ):
            solutions = es.ask()
            rewards = []
            for params in solutions:
                reward = self.rollout(params)
                rewards.append(-reward)

            es.tell(solutions, rewards)
            es.logger.add()
            es.disp()

            mean_reward = -sum(rewards) / len(rewards)
            mean_rewards.append(mean_reward)

            best_reward = -min(reward for reward in rewards)

            print(
                f"Generation {generation + 1}/{self.num_generations}, Mean Reward: {mean_reward:.4f}, Best Reward: {best_reward:.4f}"
            )

        torch.cuda.empty_cache()
        best_params = es.result.xbest
        self.controller.set_controller_parameters(best_params)
        self._plot_mean_rewards(
            mean_rewards, output_path="./mean_reward_evolution_mdn.png"
        )

        return self.controller, best_params

    def _plot_mean_rewards(
        self, mean_rewards, output_path="./mean_reward_evolution_mdn.png"
    ):

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(mean_rewards) + 1), mean_rewards, label='Mean Reward')
        plt.xlabel("Generation")
        plt.ylabel("Mean Reward")
        plt.title("Evolution of the Mean Reward Over Generations")
        plt.legend()
        plt.grid()
        plt.savefig(output_path)
        plt.close()
