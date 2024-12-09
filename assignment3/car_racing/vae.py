import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels,
        latent_dim=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super(Encoder, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(input_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.fc_mu = nn.Linear(1024, self.latent_dim)
        self.fc_sigma = nn.Linear(1024, self.latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_sigma(x)

        ##riparametrization trick
        sigma = log_var.exp()  ##to keep sigma > 0
        epsilon = torch.rand_like(sigma).to(self.device)
        z = epsilon.mul(sigma).add_(mu)

        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(
        self,
        input_channels,
        latent_dim=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, input_channels, kernel_size=6, stride=2)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 1024, 1, 1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

        return x


class VAE(nn.Module):
    def __init__(
        self,
        input_channels,
        latent_dim=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim, device=device)
        self.decoder = Decoder(input_channels, latent_dim, device=device)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var

    def loss_function(self, reconstruction, x, mu, log_var):
        bce = F.mse_loss(reconstruction, x, reduction='sum')

        kld = -0.5 * torch.sum(1 + 2 * log_var - mu.pow(2) - (2 * log_var).exp())

        return bce + kld


# if __name__ == "__main__":
#     import gymnasium as gym
#     import torch
#     import torchvision.transforms as T
#     import matplotlib.pyplot as plt

#     env = gym.make("CarRacing-v2", render_mode="rgb_array")
#     obs, info = env.reset()

#     transform = T.Compose([
#         T.ToTensor(),  # Converti in tensore (C, H, W)
#         T.Resize((64, 64)),  # Ridimensiona a 64x64
#         T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizza tra -1 e 1
#     ])
#     # plt.title("Osservazione grezza")
#     # plt.imshow(obs)
#     # plt.show()
#     obs_tensor = transform(obs).unsqueeze(0)  # Aggiungi dimensione batch
#     print("Forma del tensore preprocessato:", obs_tensor.shape)

#     latent_dim = 32
#     vae = VAE(obs_tensor.shape[1], latent_dim=latent_dim).to('cuda')

#     obs, info = env.reset()

#     for _ in range(10):  # Testa per 100 step
#         obs_tensor = transform(obs).unsqueeze(0).to('cuda') # Preprocessing
#         reconstruction, mu, sigma = vae(obs_tensor)

#         # plt.title("Osservazione originale")
#         # plt.imshow(reconstruction[0].cpu().detach().permute(1, 2, 0).numpy())
#         # plt.show()
#         print("Mu shape:", mu.shape)
#         print("Sigma shape:", sigma.shape)

#         action = env.action_space.sample()
#         obs, reward, done, truncated, info = env.step(action)
#         obs_tensor.cpu().detach().numpy()
#         if done or truncated:
#             break


#     env.close()
#     vae.to('cpu') # Sposta il modello sulla CPU
#     torch.cuda.empty_cache()  # Libera la memoria della GPU
