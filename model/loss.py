import torch
import torch.nn as nn
import torch.optim as optim


# 定义自编码器和距离矩阵损失
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class DistanceMatrixLoss(nn.Module):
    def __init__(self):
        super(DistanceMatrixLoss, self).__init__()

    def forward(self, x, y):
        dist_x = torch.cdist(x, x, p=2.0)
        dist_y = torch.cdist(y, y, p=2.0)
        loss = torch.norm(dist_x - dist_y)
        return loss


# 示例数据和模型
input_dim = 32
latent_dim = 4
model = Autoencoder(input_dim, latent_dim)
distance_matrix_loss = DistanceMatrixLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
x = torch.randn(100, input_dim)  # 100个样本

# 示例训练过程
num_epochs = 10
for epoch in range(num_epochs):
    # 假设这里有训练数据 x
    optimizer.zero_grad()
    reconstructed, latent = model(x)
    reconstruction_loss = nn.MSELoss()(reconstructed, x)
    distance_loss = distance_matrix_loss(latent, x)  # 计算距离矩阵损失
    total_loss = distance_loss
    total_loss.backward()
    optimizer.step()
    print(
        f"Epoch {epoch+1}, Reconstruction Loss: {reconstruction_loss.item()}, Distance Loss: {distance_loss.item()}"
    )
