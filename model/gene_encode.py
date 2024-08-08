import torch
import torch.nn as nn
from .triplet_loss import TripletLoss
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Encoder(nn.Module):
    def __init__(self, input_sizes, n_hidden, dim_z):
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList(
            [self.build_encoder(input_size, n_hidden) for input_size in input_sizes]
        )
        self.fc1 = nn.Linear(n_hidden, dim_z)

    def build_encoder(self, input_size, n_hidden):
        return nn.Sequential(
            nn.Linear(input_size, n_hidden, bias=True),
            nn.ELU(),
        )

    def forward(self, input_data_list):
        encoded_data_list = [
            encoder(input_data)
            for encoder, input_data in zip(self.encoders, input_data_list)
        ]
        latent_list = [self.fc1(encoded_data) for encoded_data in encoded_data_list]
        return latent_list


class SinglecellNet(nn.Module):
    def __init__(self, input_sizes, n_hidden, dim_z):
        super(SinglecellNet, self).__init__()
        self.encoder = Encoder(input_sizes, n_hidden, dim_z)
        self.decoders = nn.ModuleList(
            [self.build_decoder(dim_z, output_size) for output_size in input_sizes]
        )
        self.triplet_margin = 0

    def build_decoder(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, 256, bias=True),
            nn.ELU(),
            nn.Linear(256, output_size, bias=True),
        )

    def forward(self, input_data_list, cluster_label_list):
        latent_list = self.encoder(input_data_list)
        reconstructed_data_list = [
            decoder(latent) for decoder, latent in zip(self.decoders, latent_list)
        ]

        # 计算重构损失和稀疏惩罚，这可能需要进一步定制
        total_reconstruction_loss = [
            torch.norm((reconstructed_data - input_data))
            for input_data, reconstructed_data in zip(
                input_data_list, reconstructed_data_list
            )
        ]

        total_reconstruction_loss = sum(total_reconstruction_loss)
        # print('total_reconstruction_loss',total_reconstruction_loss)
        total_sparse_penalty = [
            torch.sqrt(torch.sum(torch.sum(decoder[0].weight ** 2, dim=1)))
            for decoder in self.decoders
        ]
        total_sparse_penalty = sum(total_sparse_penalty)

        # trip_loss_x = [TripletLoss(margin=self.triplet_margin)(latent, label) \
        #    for latent,label in zip(latent_list,cluster_label_list)]
        # total_trip_loss_x = sum(trip_loss_x)
        total_trip_loss_x = torch.tensor([0])

        return (
            latent_list,
            reconstructed_data_list,
            total_reconstruction_loss,
            total_sparse_penalty,
            total_trip_loss_x,
        )
