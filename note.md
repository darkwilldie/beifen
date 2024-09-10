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
```python
X_std_batch torch.Size([3366, 1])
Y_std_batch torch.Size([3366, 1])
corr_matrix torch.Size([3366, 1, 1])
out torch.Size([3366, 1, 1])
```

TODO
- [] 找出每个epoch进行赋值，最后效果差的原因
- [] 验证cos sim和cca等价
- [] 整合代码，纵向变成横向
- [] 检验计算cca时用不用image信息的影响

DONE
- [x] 用tangram每个epoch的损失画折线图
- [x] 整合了后处理代码，现在可以用命令行参数统一指定。
- [x] 检查出了两处代码错误，现在的loss曲线平滑

发现代码的一个重大错误
```python
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)
```
在升维的时候应该变为
```python
    X = X - X.mean(dim=1).unsqueeze(1)
    Y = Y - Y.mean(dim=1).unsqueeze(1)
```
改过来之后的变化有：
1. 效果变好
2. 损失曲线变平滑，在70 epoch之后损失下降速度指数级变快，迅速从1000级别变成120epoch的e+11级别。
3. 推测由于1.，在150个epoch前，在cca内部炸成nan
