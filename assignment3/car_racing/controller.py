import torch
import torch.nn as nn

torch.manual_seed(42)


class Controller(nn.Module):
    """Our controller for the car racing environment. a_t = W[z_t, h_t] + b"""

    def __init__(self, latent_dim, hidden_dim):
        super(Controller, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dz = latent_dim
        self.dh = hidden_dim
        self.n_action = 3
        self.fc = nn.Linear(self.dz + self.dh, self.n_action).to(torch.float32)

    def forward(self, z, h):

        x = torch.cat([z, h], dim=-1)
        x = torch.tanh(self.fc(x))

        return x

    def set_controller_parameters(self, params):
        num_weights = self.fc.weight.numel()
        num_biases = self.fc.bias.numel()
        weight_shape = self.fc.weight.shape

        if len(params) != num_weights + num_biases:
            raise ValueError(
                f"Expected {num_weights + num_biases} parameters, got {len(params)}."
            )
        with torch.no_grad():
            self.fc.weight.data = torch.tensor(
                params[:num_weights], dtype=torch.float32
            ).view(weight_shape)

            self.fc.bias = torch.nn.Parameter(
                torch.tensor(params[num_weights:], dtype=torch.float32)
            )
