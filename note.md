==test_1000需要先运行完后处理，才能在本机绘图==
```python
# S是单细胞数据，维度1，512是基因，9000是测序细胞个数
# G是空转数据，43757是spot数
S torch.Size([9000, 512])
G torch.Size([43757, 512])
image torch.Size([43757, 512])
# 损失中的G实际上是spot_feature
spot_feature torch.Size([512, 43757, 2])
# M是形状为(细胞数，spot数)的正态分布张量
M.shape torch.Size([9000, 43757])
S.shape torch.Size([9000, 512])

M_block.shape torch.Size([9000, 43757])
S_batch.shape torch.Size([9000, 512])
G_pred.shape torch.Size([43757, 512])
G.shape torch.Size([512, 43757, 2])
sc_torch torch.Size([43757, 512, 1])
st_torch torch.Size([43757, 512, 2])
X_c torch.Size([43757, 512, 1])
Y_c torch.Size([43757, 512, 1])
X_std_batch torch.Size([3000, 1])
Y_std_batch torch.Size([3000, 1])
cov_matrix torch.Size([43757, 1, 1])

time of loss: 6.1288511753082275
time of backward: 37.786664724349976
```
```python
self.S = torch.tensor(S, device=device, dtype=torch.float32)
self.G = torch.tensor(G, device=device, dtype=torch.float32)
self.image = torch.tensor(image, device=device, dtype=torch.float32)

# 使用 torch.cat 在第一个维度上堆叠它们

self.spot_feature = torch.cat(
    (self.G.unsqueeze(0), self.image.unsqueeze(0)), dim=0
).permute(2, 1, 0)

```