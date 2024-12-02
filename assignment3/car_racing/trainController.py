import torch
import numpy as np
import multiprocessing as mp
from cma import CMAEvolutionStrategy
import gymnasium as gym

from MDNLSTM import MDNLSTM_Controller

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def rollout(controller, vae, mdn_lstm:MDNLSTM_Controller, env=gym.make('CarRacing-v2'), steps=500):
    obs = env.reset()
    h = (torch.zeros(1, mdn_lstm.hidden_dim).to(device),
         torch.zeros(1, mdn_lstm.hidden_dim).to(device))
    total_reward = 0
    done = False
    for _ in range(steps):
        if done:
            break
        obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        z, _, _ = vae(obs)
        a = controller(z, h)
        a.detach().cpu().numpy()
        obs, reward, truncated, terminated, _ = env.step(a)
        done = truncated or terminated

        total_reward += reward
        _, _, _, h = mdn_lstm(z, a, h)

    return total_reward    


import multiprocessing as mp
import torch
import numpy as np
from cma import CMAEvolutionStrategy

def slave_routine(controller, vae, mdn_lstm, env_name, param_queue, result_queue, device):
    """
    Processo parallelo per eseguire rollouts con i parametri specificati.
    """
    env = gym.make(env_name)
    while True:
        params = param_queue.get()
        if params is None:  
            break

        torch.nn.utils.vector_to_parameters(torch.tensor(params).to(device), controller.parameters())

        rewards = []
        for _ in range(16):
            rewards.append(rollout(controller, vae, mdn_lstm, env))
        avg_reward = np.mean(rewards)

        result_queue.put(avg_reward)


def train_controller_with_cmaes_parallel(controller, vae, mdn_lstm, env_name='CarRacing-v2',
                                         num_iterations=100, population_size=64, max_steps=1000, num_workers=None):
    """
    Addestra il controller utilizzando CMA-ES con rollout paralleli, con numero di worker configurabile.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    param_vector = torch.nn.utils.parameters_to_vector(controller.parameters()).detach().cpu().numpy()
    cma_es = CMAEvolutionStrategy(param_vector, 0.5, {'popsize': population_size})

    if num_workers is None:
        num_workers = mp.cpu_count()

    param_queue = mp.Queue()
    result_queue = mp.Queue()


    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=slave_routine, args=(controller, vae, mdn_lstm, env_name, param_queue, result_queue, device))
        p.start()
        processes.append(p)

    for iteration in range(num_iterations):
        solutions = cma_es.ask()  

        for solution in solutions:
            param_queue.put(solution)

        rewards = []
        for _ in range(len(solutions)):
            rewards.append(result_queue.get())

        cma_es.tell(solutions, [-r for r in rewards])


        print(f"Iterazione {iteration + 1}/{num_iterations}, Ricompensa media: {np.mean(rewards):.2f}, Ricompensa migliore: {np.max(rewards):.2f}")

        if (iteration + 1) % 25 == 0:
            best_params = cma_es.best.x
            torch.nn.utils.vector_to_parameters(torch.tensor(best_params).to(device), controller.parameters())
            rewards = [rollout(controller, vae, mdn_lstm, gym.make(env_name)) for _ in range(1024)]
            avg_best_reward = np.mean(rewards)
            print(f"Generazione {iteration + 1}, Ricompensa media del miglior agente su 1024 rollout: {avg_best_reward:.2f}")
            torch.save(controller.state_dict(), f"controller_best_{iteration + 1}.pth")

    for _ in processes:
        param_queue.put(None)
    for p in processes:
        p.join()

    return controller



# def train_controller(controller, vae, mdn_lstm, env, num_rollouts=10, steps=500):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     param_vector = torch.nn.utils.parameters_to_vector(controller.parameters()).detach().cpu().numpy()
#     cma_es = CMAEvolutionStrategy(param_vector, 0.5, {'popsize': population_size})





# if __name__ == "__main__":
#     from vae import VAE
#     vae = VAE(3, 32)
#     vae.cuda()
#     rollout(None, vae)
