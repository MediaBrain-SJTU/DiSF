from sklearn.manifold import TSNE

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# 步骤1: 特征归一化到单位球面上
# Step 1: Feature normalization to the unit sphere  
def normalize_features_to_sphere(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / norms
    return normalized_features

# 步骤2: 生成单位球体的表面坐标
# Step 2: Generate the surface coordinates of the unit sphere  
def create_sphere():
    phi, theta = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

# 步骤3: 绘制单位球体和特征点
# Step 3: Draw the unit sphere and feature points  
def plot_sphere_and_features(features):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制球体
    x, y, z = create_sphere()
    ax.plot_surface(x, y, z, color='c', alpha=0.3, rstride=5, cstride=5)

    # 绘制特征点
    normalized_features = normalize_features_to_sphere(features)
    #print(normalized_features.shape)
    ax.scatter(normalized_features[:, 0], normalized_features[:, 1], normalized_features[:, 2], color='r', s=10)

    # 设置坐标轴范围
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.show()
    plt.savefig("./3d.pdf")

# 读取本地特征
# Read local features  

features=torch.tensor(np.load("./dsir.npy"))

#features = np.random.rand(10, 3) - 0.5  # 生成随机特征并偏移到中心

tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=750)
tsne_results = tsne.fit_transform(features)

# 选择部分样本做可视化
# Select some samples for visualization  

tsne_results=tsne_results[1000:1300]

plot_sphere_and_features(tsne_results)
