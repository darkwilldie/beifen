import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, input_size, n_hidden, dim_z):
        super(ImageEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, dim_z, bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_z, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, input_size, bias=True),
        )

    def forward(self, input_data):
        latent = self.encoder(input_data)
        reconstructed_data = self.decoder(latent)
        reconstruction_losses = torch.norm(
            torch.sign(input_data) * (reconstructed_data - input_data)
        )

        total_sparse_penalty = torch.sqrt(
            torch.sum(torch.sum(self.decoder[0].weight ** 2, dim=1))
        )

        return latent, reconstructed_data, reconstruction_losses, total_sparse_penalty
