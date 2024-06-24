# 从头开始构建满足所有要求的代码，包括插值、颜色平滑以及保存图像
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
# 创建更平滑的数据插值函数
# def smooth_data(kx_sparse, ky_sparse, E_sparse, k_range, method='cubic', grid_num=500):
#     kx_fine = np.linspace(-1*k_range, k_range, grid_num)
#     ky_fine = np.linspace(-1*k_range, k_range, grid_num)
#     kx_fine, ky_fine = np.meshgrid(kx_fine, ky_fine)
#     E_fine = griddata((kx_sparse, ky_sparse), E_sparse, (kx_fine, ky_fine), method=method)
#     return kx_fine, ky_fine, E_fine
def smooth_data(kx_sparse, ky_sparse, E_sparse, k_range, method='cubic', grid_num=500):
    grid_x, grid_y = np.mgrid[np.min(kx_sparse):np.max(kx_sparse):500j, np.min(kx_sparse):np.max(kx_sparse):500j]
    grid_z = griddata((kx_sparse, ky_sparse), E_sparse, (grid_x, grid_y), method='cubic')
    return grid_x, grid_y, grid_z.T
def draw(kx,ky,weight,k_range,figname1):
    fig, ax = plt.subplots(figsize =(7,6))
    im = ax.imshow(weight, extent=(-1*k_range,k_range,-1*k_range,k_range), cmap = 'viridis', interpolation='gaussian', origin='lower')
    cbar = fig.colorbar(im, ax=ax, orientation = 'vertical')
    cbar.set_label('DOS')
    ax.set_title(figname1)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    # 保存图片
    current_path = os.getcwd()
    folder_name = '/fig/'
    folder_path = os.path.join(current_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    time = datetime.datetime.now().strftime('%m-%d %H_%M')  # 括号里('%Y-%m-%d %H_%M_%S') 年月日，时分秒
    figname = figname1
    filetype = '.png'
    plt.savefig(folder_path + figname + time + filetype,transparent=True, format='png',bbox_inches='tight',pad_inches=0.0, dpi=300)
    # 显示图片
    plt.show()
    plt.clf()
# 模拟稀疏数据
'''
kx_sparse = np.random.uniform(-1, 1, 100)
ky_sparse = np.random.uniform(-1, 1, 100)
E_sparse = np.sin(np.sqrt(kx_sparse**2 + ky_sparse**2))

# 插值数据
kx_fine, ky_fine, E_fine = smooth_data(kx_sparse, ky_sparse, E_sparse)

# 绘制图像
fig, ax = plt.subplots(figsize=(7, 6))

# 绘制平滑颜色变化的等能面图
im = ax.imshow(E_fine, extent=(-1, 1, -1, 1), cmap='viridis', interpolation='bilinear', origin='lower')

# 添加颜色条
cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.set_label('Energy E')

# 设置标题和轴标签
ax.set_title('Interpolated Energy Contour')
ax.set_xlabel('kx')
ax.set_ylabel('ky')

# 保存图片
current_path = os.getcwd()
#parent_path = os.path.abspath(os.path.join(current_path, '..'))
folder_name = 'fig'
folder_path = os.path.join(current_path, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
time = datetime.datetime.now().strftime('%m-%d %H_%M') # 括号里('%Y-%m-%d %H_%M_%S') 年月日，时分秒
figname = '/energy_contour'
filetype = '.png'
plt.savefig(folder_path + figname + time + filetype, format='png', dpi=300)

# 显示图片
plt.show()

# 输出保存的文件路径
# print('/mnt/data/energy_contour.png')
'''