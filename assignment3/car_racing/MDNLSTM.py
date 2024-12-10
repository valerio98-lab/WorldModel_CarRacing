import torch
import torch.nn as nn
import torch.nn.functional as F

"""Memory component of the world model". An RNN + MDN"""


class LSTM(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True).to(self.device)

    def forward(self, x, hidden_state=None):
        x = x.to(self.device)  # Sposta l'input sul dispositivo corretto
        if hidden_state is None:
            hidden_state = (
                torch.zeros(1, x.size(0), self.hidden_dim, device=self.device),
                torch.zeros(1, x.size(0), self.hidden_dim, device=self.device),
            )

        # Passa attraverso la LSTM
        # stampa il device della lstm
        x, hidden_state = self.lstm(x, hidden_state)
        return x, hidden_state


class MDN(nn.Module):
    def __init__(
        self,
        latent_dim,
        action_dim,
        hidden_dim=256,
        num_gaussians=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super(MDN, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        self.output_layer = nn.Linear(hidden_dim, (2 * latent_dim + 1) * num_gaussians)

    def forward(self, hidden_state):
        out = self.output_layer(hidden_state)

        # total_dim = self.num_gaussians * (2 * self.latent_dim + 1)
        # assert (
        #     out.size(-1) == total_dim
        # ), f"Dimension mismatch: expected {total_dim}, got {out.size(-1)}"

        batch_size, num_frames = hidden_state.size(0), hidden_state.size(1)

        offset = self.num_gaussians * self.latent_dim

        alpha = out[:, :, : self.num_gaussians]
        mu = out[:, :, self.num_gaussians : offset + self.num_gaussians]
        sigma = out[:, :, offset + self.num_gaussians :]

        mu = mu.view(batch_size, num_frames, self.num_gaussians, self.latent_dim)
        sigma = sigma.view(batch_size, num_frames, self.num_gaussians, self.latent_dim)

        alpha = F.log_softmax(alpha, dim=-1)
        sigma = torch.exp(sigma)

        # r = out[:, :, -2]
        # d = out[:, :, -1]

        return alpha, mu, sigma


class MDNLSTM(nn.Module):
    """LSTM + MDN"""

    def __init__(
        self,
        latent_dim,
        action_dim,
        hidden_dim,
        num_gaussians=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super(MDNLSTM, self).__init__()
        self.device = device
        self.input_dim = latent_dim + action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_gaussians = num_gaussians

        self.lstm = LSTM(self.input_dim, hidden_dim, device=self.device)
        self.mdn = MDN(
            latent_dim, action_dim, hidden_dim, num_gaussians, device=self.device
        )

    def forward(self, latent_vector, action_vector, hidden_state=None):
        # Sposta gli input sul dispositivo corretto
        latent_vector = latent_vector.to(self.device)
        action_vector = action_vector.to(self.device)
        # Concatenazione degli input
        x = torch.cat((latent_vector, action_vector), dim=-1)  # [batch_size, input_dim]
        x = x.unsqueeze(0) if len(x.shape) == 2 else x  # Aggiunge seq_len=1 se necessario
        # Passaggio attraverso la LSTM
        out, hidden_state = self.lstm(x, hidden_state)
        # Passaggio attraverso l'MDN
        alpha, mu, sigma = self.mdn(out)
        return alpha, mu, sigma, hidden_state

    def mdn_loss(self, alpha, sigma, mu, target, eps=1e-8):
        target = target.unsqueeze(2)
        target = target.expand(-1, -1, mu.size(2), -1)

        assert target.shape == mu.shape, "Mismatch in target shape"

        m = torch.distributions.Normal(loc=mu, scale=sigma)
        log_prob = m.log_prob(target)
        log_prob = alpha + log_prob.sum(dim=-1)
        max_log_prob = log_prob.max(dim=-1, keepdim=True)[0]
        log_prob = log_prob - max_log_prob

        prob = torch.exp(log_prob)
        prob = prob.sum(dim=-1)

        out_loss = max_log_prob.squeeze() + torch.log(prob + eps)

        return -out_loss.mean()

    def sample(self, alpha, mu, sigma):
        # Campiona la componente della miscela
        categorical = torch.distributions.Categorical(alpha)
        component = categorical.sample()  # Shape: [batch_size, num_frames]

        # Usa torch.gather per selezionare le componenti corrette
        batch_size, num_frames, num_gaussians, latent_dim = mu.shape
        component = component.unsqueeze(-1).unsqueeze(
            -1
        )  # Shape: [batch_size, num_frames, 1, 1]

        # Seleziona mu e sigma
        chosen_mu = torch.gather(
            mu, 2, component.expand(batch_size, num_frames, 1, latent_dim)
        ).squeeze(2)
        chosen_sigma = torch.gather(
            sigma, 2, component.expand(batch_size, num_frames, 1, latent_dim)
        ).squeeze(2)

        # Campiona dalla normale
        normal = torch.distributions.Normal(chosen_mu, chosen_sigma)
        sample = normal.sample()  # Shape: [batch_size, num_frames, latent_dim]

        return sample


# class MDNLSTM_Controller(nn.Module):
#     """In order to train successfully the controller and in accordance with the paper
#     we need to have an explicit management of the hidden state rather than an implicit as in the previous model


#     The logic could be plugged in the previous model with a slight modification of the forward method
#     but for the sake of clarity we keep it separate"""

#     def __init__(self, latent_dim, action_dim, hidden_dim, num_gaussians=5):
#         super(MDNLSTM_Controller, self).__init__()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.latent_dim = latent_dim
#         self.action_dim = action_dim
#         self.hidden_dim = hidden_dim
#         self.num_gaussians = num_gaussians

#         self.lstm = LSTM(latent_dim + action_dim, hidden_dim).to(self.device)
#         self.mdn = MDN(latent_dim, action_dim, hidden_dim, num_gaussians).to(self.device)

#     def forward(self, latent_vector, action_vector, hidden_state=None):
#         latent_vector = latent_vector.to(self.device)
#         action_vector = action_vector.unsqueeze(0).to(self.device)

#         x = torch.cat((latent_vector, action_vector), dim=-1)
#         x = x.unsqueeze(0) if len(x.shape) == 2 else x

#         out, hidden_state = self.lstm(x, hidden_state)
#         alpha, mu, sigma = self.mdn(out)

#         return alpha, mu, sigma, hidden_state


# if __name__ == "__main__":
#     latent_dim = 32
#     action_dim = 3
#     hidden_dim = 256
#     num_gaussians = 5

#     model = MDNLSTM(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim, num_gaussians=num_gaussians)

#     z_t = torch.randn(1, latent_dim)
#     a_t = torch.randn(1, action_dim)
#     hidden_state = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))

#     # Forward pass
#     pi, mu, sigma, hidden_state = model(z_t, a_t)

#     # Output
#     print("Pesi della miscela (pi):", pi.shape)  # (batch_size, num_gaussians)
#     print("Medie (mu):", mu.shape)              # (batch_size, num_gaussians, latent_dim)
#     print("Deviazioni standard (sigma):", sigma.shape)
