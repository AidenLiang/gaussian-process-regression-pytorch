import torch
import matplotlib.pyplot as plt

# 生成带噪声的正弦波数据（GPU加速）
X = torch.linspace(0, 10, 100, device="mps").unsqueeze(1)
y = torch.sin(X) + 0.1 * torch.randn(X.shape, device="mps")

# 可视化检查（可选）
# plt.scatter(X.cpu(), y.cpu(), s=10)
# plt.title("Synthetic Training Data")
# plt.savefig("data.png")  # 保存图片备用

class RBFKernel(torch.nn.Module):
    def forward(self, X1, X2):
        dist = torch.cdist(X1, X2, p=2)  # MPS加速的欧氏距离
        return torch.exp(-0.5 * dist.pow(2))  # 固定length_scale=1.0
    
def gpr_predict(X_train, y_train, X_test, noise=1e-4):
    kernel = RBFKernel()
    K = kernel(X_train, X_train) + noise * torch.eye(X_train.shape[0], device="mps")
    
    # Cholesky分解（MPS->CPU避免数值问题）
    L = torch.linalg.cholesky(K.cpu()).to("mps")
    alpha = torch.cholesky_solve(y_train, L)
    
    K_s = kernel(X_train, X_test)
    mu = K_s.T @ alpha
    return mu.cpu()  # 返回CPU数据方便绘图

X_test = torch.linspace(0, 10, 50, device="mps").unsqueeze(1)
mu = gpr_predict(X, y, X_test)  # 测试预测