import torch 
import torch.nn as nn
import torch.nn.functional as F

"""Memory component of the world model". An RNN + MDN"""

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = (torch.zeros(1, x.size(0), self.hidden_dim).to(x.device),
                            torch.zeros(1, x.size(0), self.hidden_dim).to(x.device))
        
        x, hidden_state = self.lstm(x, hidden_state)

        return x, hidden_state


class MDN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=256, num_gaussians=5):
        super(MDN, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        self.output_layer = nn.Linear(hidden_dim, num_gaussians * (2 * latent_dim + 1))
    

    def forward(self, hidden_state):
        out = self.output_layer(hidden_state)
        
        offset = self.num_gaussians + self.num_gaussians * self.latent_dim

        alpha = out[:, :self.num_gaussians]
        mu = out[:, self.num_gaussians:offset]
        sigma = out[:, offset:]

        alpha = F.softmax(alpha.view(-1, self.num_gaussians), dim=1)
        mu = mu.view(-1, self.num_gaussians, self.latent_dim)
        sigma = torch.exp(sigma.view(-1, self.num_gaussians, self.latent_dim))

        return alpha, mu, sigma



    def mdn_loss(self, alpha, sigma, mu, target, eps=1e-8):
        target = target.unsqueeze(1).expand_as(mu)
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        log_prob = m.log_prob(target)
        log_prob = log_prob.sum(dim=2)
        log_alpha = torch.log(alpha + eps)  # Avoid log(0) disaster
        loss = -torch.logsumexp(log_alpha + log_prob, dim=1)
        return loss.mean()



class MDNLSTM(nn.Module):
    """ LSTM + MDN"""
    def __init__(self, latent_dim, action_dim, hidden_dim, num_gaussians=5):
        super(MDNLSTM, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = latent_dim + action_dim
        self.lstm = LSTM(self.input_dim, hidden_dim)
        self.mdn = MDN(latent_dim, action_dim, hidden_dim, num_gaussians)

    def forward(self, latent_vector, action_vector):
        x = torch.cat((latent_vector, action_vector), dim=-1)

        x = x.unsqueeze(0) if len(x.shape) == 2 else x

        out, hidden_state = self.lstm(x)
        alpha, mu, sigma = self.mdn(out[:, -1, :])

        return alpha, mu, sigma, hidden_state


        

if __name__ == "__main__":
    latent_dim = 32
    action_dim = 3
    hidden_dim = 256
    num_gaussians = 5

    # Modello
    model = MDNLSTM(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim, num_gaussians=num_gaussians)

    # Input fittizio
    z_t = torch.randn(1, latent_dim)  # Stato latente corrente
    a_t = torch.randn(1, action_dim)  # Azione
    hidden_state = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))  # Stato iniziale della LSTM

    # Forward pass
    pi, mu, sigma, hidden_state = model(z_t, a_t)

    # Output
    print("Pesi della miscela (pi):", pi.shape)  # (batch_size, num_gaussians)
    print("Medie (mu):", mu.shape)              # (batch_size, num_gaussians, latent_dim)
    print("Deviazioni standard (sigma):", sigma.shape) 