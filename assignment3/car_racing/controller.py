import torch 
import torch.nn as nn

torch.manual_seed(42)


class Controller(nn.Module):
    """ Our controller for the car racing environment. a_t = W[z_t, h_t] + b """
    def __init__(self, latent_vector_dim, hidden_vector_dim):
        super(Controller, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dz = latent_vector_dim
        self.dh = hidden_vector_dim
        self.n_action = 3
        self.fc = nn.Linear(self.dz + self.dh, self.n_action) #+b implicit in the linear layer
    
    def forward(self, z, h):
        x = torch.cat((z,h), dim=1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    controller = Controller(32, 64)
    print(controller)
    z = torch.randn(1, 32)
    h = torch.randn(1, 64)
    print(controller(z, h))
    print(controller(z, h).shape)