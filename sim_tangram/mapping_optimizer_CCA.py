"""
Library for instantiating and running the optimizer for Tangram. The optimizer comes in two flavors,
which correspond to two different classes:
- Mapper: optimizer without filtering (i.e., all single cells are mapped onto space). At the end, the learned mapping
matrix M is returned.
- MapperConstrained: optimizer with filtering (i.e., only a subset of single cells are mapped onto space).
At the end, the learned mapping matrix M and the learned filter F are returned.
"""

import numpy as np
import logging
import torch
from torch.nn.functional import softmax, cosine_similarity
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import torch.linalg as LA
from tqdm import tqdm
# from model.__init__ import cos_sim
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt


def cos_sim(sc_feature, other_feature):
    sc_feature = F.normalize(sc_feature, dim=1)
    other_feature = F.normalize(other_feature, dim=1)
    cos_sim_matrix = torch.matmul(sc_feature, other_feature.T)
    # softmax_output = F.softmax(cos_sim_matrix, dim=1)
    # print(softmax_output)
    return cos_sim_matrix

def cos_sim_dig(sc_feature, other_feature):
    sc_feature = F.normalize(sc_feature, dim=1)
    other_feature = F.normalize(other_feature, dim=1)
    cos_sim_matrix = torch.sum(sc_feature * other_feature, dim=1, keepdim=True)
    assert cos_sim_matrix.shape == (sc_feature.shape[0], 1)
    return cos_sim_matrix
    
def torch_corrcoef(X, Y, batch_size=4000):
    X_demean = X - torch.mean(X, dim=1, keepdim=True)
    Y_demean = Y - torch.mean(Y, dim=1, keepdim=True)
    cov_matrix = torch.matmul(X_demean.transpose(1, 2), Y_demean) / (X.shape[1] - 1)
    # print('cov_matrix', cov_matrix.shape)
    X_std = torch.std(X_demean, dim=1)
    X_std = X_std.view(X_std.shape[0], -1)
    Y_std = torch.std(Y_demean, dim=1)
    Y_std = Y_std.view(Y_std.shape[0], -1)

    # corr_matrix = cov_matrix / torch.outer(X_std, Y_std)
    corr_matrix = torch.zeros_like(cov_matrix)
    
    num_batches = (X_std.shape[0] + batch_size - 1) // batch_size
    
    for i in (range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_std.shape[0])
        
        X_std_batch = X_std[start_idx:end_idx, :]
        Y_std_batch = Y_std[start_idx:end_idx, :]

        # out = torch.outer(X_std_batch.squeeze(1), Y_std_batch.squeeze(1))
        # print('out', out.shape)
        # print('X_std_batch', X_std_batch.shape)
        # print('Y_std_batch', Y_std_batch.shape)
        # print('corr_matrix', corr_matrix.shape)
        #! 代码写错了，out直接赋值导致循环完全没有意义
        out = torch.einsum('bi,bj->bij', X_std_batch, Y_std_batch)
        # for j in range(X_std_batch.shape[0]):
            # out = torch.outer(X_std_batch[j, :], Y_std_batch[j, :])
        # print('out', out.shape)
        # assert False

        corr_matrix[start_idx:end_idx, :] = cov_matrix[start_idx:end_idx, :] / out

    # print('corr_matrix', corr_matrix.shape)
    return corr_matrix


def torch_cca(X, Y, n_components=2, reg_param=1e-5):
    device = X.device  # 确保计算在同一设备上（GPU或CPU）

    X = X.float()
    Y = Y.float()

    # 标准化和中心化
    #! 二维计算改为三维，此处的dim应从0->1
    X = X - X.mean(dim=1).unsqueeze(1)
    Y = Y - Y.mean(dim=1).unsqueeze(1)

    n = X.size(1) - 1

    # 确保 X 和 Y 是三维矩阵
    if X.ndim != 3:
        raise ValueError(
            "X must be a 3D tensor with shape (batch_size, num_samples, num_features)"
        )
    if Y.ndim != 3:
        raise ValueError(
            "Y must be a 3D tensor with shape (batch_size, num_samples, num_features)"
        )

    cov_xx = torch.matmul(X.transpose(1, 2), X) / n + reg_param * torch.eye(
        X.size(-1), device=device
    )
    cov_yy = torch.matmul(Y.transpose(1, 2), Y) / n + reg_param * torch.eye(
        Y.size(-1), device=device
    )
    cov_xy = torch.matmul(X.transpose(1, 2), Y) / n

    # Cholesky分解
    assert torch.all(cov_xx >= 0), print(cov_xx)
    chol_xx = LA.cholesky(cov_xx)
    chol_yy = LA.cholesky(cov_yy)

    # 白化
    whitened_x = torch.empty_like(X)
    whitened_y = torch.empty_like(Y)

    # st_time1 = time.time()
    whitened_x = LA.solve_triangular(chol_xx, X.transpose(1, 2), upper=False).transpose(1, 2)
    whitened_y = LA.solve_triangular(chol_yy, Y.transpose(1, 2), upper=False).transpose(1, 2)

    # end_time1 = time.time()
    # print('time of process matrix:', end_time1 - st_time1)

    # 奇异值分解
    u, _, v = torch.svd(torch.matmul(whitened_x.transpose(1, 2), whitened_y))

    X_c = torch.matmul(whitened_x, u[:, :, :n_components])
    Y_c = torch.matmul(whitened_y, v[:, :, :n_components])

    # 使用PyTorch计算的相关系数矩阵
    # print('X_c', X_c.shape)
    # print('Y_c', Y_c.shape)

    # st_time2 = time.time()
    correlations_matrix = torch_corrcoef(X_c, Y_c)
    # end_time2 = time.time()
    # print('time of corrcoef:', end_time2 - st_time2)

    return X_c, Y_c, correlations_matrix


# sc_torch.shape = (43757, 512)
# st_torch.shape = (512, 43757, 2)
def calculate_correlations(sc_torch, st_torch):
    sum_cos_sim1 = 0

    # print('&&sc_torch', sc_torch.shape)
    # print('&&st_torch', st_torch.shape)
    # time1 = time.time()
    # # 余弦相似度
    cos_sim1 = cos_sim_dig(sc_torch, st_torch[:,:,0].t()).unsqueeze(-1)
    # print('cos_sim1', cos_sim1.shape)
    # sum_cos_sim = cos_sim(sc_torch, st_torch.permute(1, 0, 2)[:,:,0])
    # time2 = time.time()
    # print('time of cos_sim:', time2 - time1)

    st_torch = st_torch.permute(1, 0, 2)
    sc_torch = sc_torch.unsqueeze(-1)
    # print('sc_torch', sc_torch.shape)
    # print('st_torch', st_torch.shape)
    # start_time = time.time()
    X_c_torch, Y_c_torch, torch_correlations = torch_cca(
        sc_torch, st_torch, n_components=1
    )
    # print('torch_correlations', torch_correlations.shape)
    assert torch_correlations.shape == cos_sim1.shape

    torch_correlations = torch.sign(cos_sim1) * torch.abs(torch_correlations)

    # end_time = time.time()
    # print('time of torch_cca:', end_time - start_time)
    torch_correlations = torch_correlations.sum()

    sum_cos_sim1 = sum_cos_sim1 + torch_correlations  # 取典型相关系数
    return sum_cos_sim1


class Mapper:
    """
    Allows instantiating and running the optimizer for Tangram, without filtering.
    Once instantiated, the optimizer is run with the 'train' method, which also returns the mapping result.
    """

    def __init__(
        self,
        S,
        G,
        image,
        lambda_g1=1.0,
        lambda_d=0,
        lambda_g2=0,
        lambda_r=0,
        device="cpu",
        random_state=0,
    ):
        """
        Instantiate the Tangram optimizer (without filtering).
        """
        # print(device)
        # assert False

        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.G = torch.tensor(G, device=device, dtype=torch.float32)
        self.image = torch.tensor(image, device=device, dtype=torch.float32)

        # 使用 torch.cat 在第一个维度上堆叠它们

        self.spot_feature = torch.cat(
            (self.G.unsqueeze(0), self.image.unsqueeze(0)), dim=0
        ).permute(2, 1, 0)

        print('$'*50)
        print('S', self.S.shape)
        print('G', self.G.shape)
        print('image', self.image.shape)
        print('spot_feature', self.spot_feature.shape)

        self.lambda_g1 = lambda_g1
        self.lambda_d = lambda_d
        self.lambda_g2 = lambda_g2
        self.lambda_r = lambda_r
        self.random_state = random_state
        np.random.seed(seed=self.random_state)

        self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        self.M = torch.tensor(
            self.M, device=device, requires_grad=True, dtype=torch.float32
        )
        print('M', self.M.shape)
        # assert False
        # self.M = softmax(self.M, dim=1)

    def _loss_fn(self, S_batch, G_with_image, M_block):
        # M_probs = softmax(self.M, dim=1)
        G_pred = torch.matmul(M_block.t(), S_batch)
        # M.shape = (9000, 43757)
        # S.shape = (9000, 512)
        # G_pred.shape = (43757, 512)
        # G_with_image.shape = (512, 43757, 2)
        # 计算预测值和真实值之间的相关性（gv_term = (43757, 1, 1)
        # ! 目前没有用image信息
        gv_term = self.lambda_g1 * calculate_correlations(G_pred, G_with_image)
        # # 计算cos sim(shape ([9000, 43757]))
        # sum_cos_sim = cos_sim(S_batch, G_with_image[:,:,0].t())
        # print('sum_cos_sim', sum_cos_sim.shape)
        # print('M_block', M_block.shape)
        # assert M_block.shape == sum_cos_sim.shape
        # #? 赋符号（是给M_block赋符号，而非预测值和真实值之间的相关性），这一步是否合理？
        # sign_a = torch.sign(sum_cos_sim)
        # # M_block.data = torch.sign(sum_cos_sim) * torch.abs(M_block)
        # M_block = sign_a * torch.abs(M_block)

        print(gv_term)
        return -gv_term

    def train(self, num_epochs, learning_rate=0.1, print_each=100, batch_size=8):

        loss_plot = []

        # torch_path1 = f"section2_cos.pt"
        # # torch_path1 = f"E:/Omics/beifen/test_cossim/original_sc_st_Mouse_brain.pt"
        # cos_sim = torch.load(torch_path1, map_location="cpu")
        # sign_a = torch.sign(cos_sim).float().detach()
        
        torch.manual_seed(self.random_state)
        optimizer = torch.optim.Adam([self.M], lr=learning_rate)
        # print('M.shape', self.M.shape)
        # print('S.shape', self.S.shape)
        num_batches = (self.S.shape[0] + batch_size - 1) // batch_size
        # s cell 100*50  g spot 40*50
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            for i in (range(num_batches)):

                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                S_batch = self.S[start_idx:end_idx]
                M_block = self.M[start_idx:end_idx]

                #? 是否应该先赋符号，再softmax？
                M_block = softmax(M_block, dim=1)
                # 计算cos sim(shape ([9000, 43757]))
                #! 这里的S和spot_feature每个epoch都不变
                # sum_cos_sim = cos_sim(S_batch, self.spot_feature[:,:,0].t())
                # print('sum_cos_sim', sum_cos_sim.shape)
                # print('M_block', M_block.shape)
                # assert M_block.shape == sum_cos_sim.shape
                #? 赋符号（是给M_block赋符号，而非预测值和真实值之间的相关性），这一步是否合理？
                # sign_a = torch.sign(sum_cos_sim)
                # M_block.data = torch.sign(sum_cos_sim) * torch.abs(M_block)
                # M_block = sign_a * torch.abs(M_block)


                # 计算损失
                # start_time = time.time()
                loss = self._loss_fn(S_batch, self.spot_feature, M_block)

                # end_time1 = time.time()
                # print('time of loss:', end_time1 - start_time)
                # print(loss.shape)
                loss_plot.append(loss.detach().cpu().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # end_time2 = time.time()
                # print('time of backward:', end_time2 - end_time1)

            if epoch % print_each == 0:
                print(f"Epoch {epoch}: Loss {epoch_loss / num_batches}")

        # 画出loss_plot的折线图
        plt.plot(loss_plot)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.savefig('loss_plot.png')
        
        with torch.no_grad():
            # self.M = sign_a * torch.abs(self.M)
            output = softmax(self.M, dim=1)
            return output.cpu().detach().numpy()
        
            
