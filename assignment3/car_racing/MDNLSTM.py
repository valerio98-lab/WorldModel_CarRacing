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
        
        total_dim = self.num_gaussians * (2 * self.latent_dim + 1)
        assert out.size(-1) == total_dim, f"Dimension mismatch: expected {total_dim}, got {out.size(-1)}"

        batch_size, num_frames = hidden_state.size(0), hidden_state.size(1)

        alpha = out[:, :, :self.num_gaussians] 
        mu = out[:, :, self.num_gaussians:self.num_gaussians + self.num_gaussians * self.latent_dim]
        sigma = out[:, :, self.num_gaussians + self.num_gaussians * self.latent_dim:]


        mu = mu.view(batch_size, num_frames, self.num_gaussians, self.latent_dim)  
        sigma = sigma.view(batch_size, num_frames, self.num_gaussians, self.latent_dim) 

        alpha = F.softmax(alpha, dim=-1)  
        sigma = torch.exp(sigma)  

        return alpha, mu, sigma

    


class MDNLSTM(nn.Module):
    """ LSTM + MDN"""
    def __init__(self, latent_dim, action_dim, hidden_dim, num_gaussians=5):
        super(MDNLSTM, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = latent_dim + action_dim
        self.lstm = LSTM(self.input_dim, hidden_dim).to(self.device)
        self.mdn = MDN(latent_dim, action_dim, hidden_dim, num_gaussians).to(self.device)


    def forward(self, latent_vector, action_vector):
        latent_vector = latent_vector.to(self.device)
        action_vector = action_vector.to(self.device)

        x = torch.cat((latent_vector, action_vector), dim=-1)
        x = x.unsqueeze(0) if len(x.shape) == 2 else x

        out, hidden_state = self.lstm(x)
        alpha, mu, sigma = self.mdn(out)

        return alpha, mu, sigma, hidden_state
    

    def mdn_loss(self, alpha, sigma, mu, target, eps=1e-8):
        target = target.unsqueeze(2)
        target = target.expand(-1, -1, mu.size(2), -1)

        assert target.shape == mu.shape, "Mismatch in target shape"

        m = torch.distributions.Normal(loc=mu, scale=sigma)
        log_prob = m.log_prob(target)
        log_prob = log_prob.sum(dim=-1)
        log_alpha = torch.log(alpha + eps)  
        loss = -torch.logsumexp(log_alpha + log_prob, dim=-1)
        return loss.mean()


class MDNLSTM_Controller(nn.Module):
    """ In order to train successfully the controller and in accordance with the paper 
    we need to have an explicit management of the hidden state rather than an implicit as in the previous model
    
    
    The logic could be plugged in the previous model with a slight modification of the forward method
    but for the sake of clarity we keep it separate"""

    def __init__(self,
                 latent_dim,
                 action_dim,
                 hidden_dim,
                 num_gaussians=5):
        super(MDNLSTM_Controller, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians

        self.lstm = LSTM(latent_dim + action_dim, hidden_dim).to(self.device)
        self.mdn = MDN(latent_dim, action_dim, hidden_dim, num_gaussians).to(self.device)


    def forward(self, latent_vector, action_vector, hidden_state=None):
        latent_vector = latent_vector.to(self.device)
        action_vector = action_vector.unsqueeze(0).to(self.device)

        x = torch.cat((latent_vector, action_vector), dim=-1)
        x = x.unsqueeze(0) if len(x.shape) == 2 else x

        out, hidden_state = self.lstm(x, hidden_state)
        alpha, mu, sigma = self.mdn(out)

        return alpha, mu, sigma, hidden_state





        

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