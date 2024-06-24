# First, let's load the data from both files to understand their format and how they can be merged for plotting.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# Load kx and ky values
kx_ky_path = r"D:\Users\Desktop\计算\画图实验\kx ky.txt"
kx, ky = np.loadtxt(kx_ky_path, usecols=(0, 1), unpack=True)
k_points = np.vstack([kx,ky,np.ones(len(kx))]).T
# Load effective mass ratios
mass_eff_ratio_path = r"D:\Users\Desktop\计算\画图实验\mass_eff_ratio.dat"
#mass_eff_ratio_path =r"D:\Users\Desktop\计算\画图实验\mass_eff_ratio_val_max.txt"
mass_eff_ratio = np.loadtxt(mass_eff_ratio_path)


# We will use only the second column from the mass_eff_ratio.dat for plotting, as it represents the effective mass ratios.
# Extracting the effective mass ratio values
mass_eff_ratio_values = mass_eff_ratio[:, 1]
mass_eff_ratio_values_all=mass_eff_ratio_values[:]
for i in range(1,12):
    mass_eff_ratio_values_all = np.append(mass_eff_ratio_values_all,mass_eff_ratio_values)
# Now, let's prepare the plots.

def convert_to_cartesian(k_points, a=1, a_abs=3.98E-9, c=1):
    # 倒格矢基矢在笛卡尔坐标系中的表示
    b1 = np.array([2 * np.pi / a, 2 * np.pi / (a * np.sqrt(3)), 0])
    b2 = np.array([0, 4 * np.pi / (a * np.sqrt(3)), 0])
    b1 = b1 / (2 * np.pi)
    b2 = b2 / (2 * np.pi)
    # 转换所有K点
    cartesian_coords = []
    cartesian_coords_abs = []
    for kx, ky, kz in k_points:
        cartesian_k = kx * b1 + ky * b2  # 这里忽略kz因为我们关注于x-y平面
        cartesian_coords.append((cartesian_k[0], cartesian_k[1]))
        cartesian_coords_abs.append(
            (2 * np.pi * cartesian_k[0] / a_abs, 2 * np.pi * cartesian_k[1] / a_abs))
    #-r和-a分别为绝对坐标和相对坐标
    kx_r = [ki[0] for ki in cartesian_coords]
    kx_a = [ki[0] for ki in k_points]
    ky_r = [ki[1] for ki in cartesian_coords]
    ky_a = [ki[1] for ki in k_points]
    #kz_r = [ki[2] for ki in cartesian_coords]
    #kz_a = [ki[2] for ki in k_points]
    fig =plt.figure()
    ax_r = fig.add_subplot(121, projection='3d')
    ax_r.scatter(kx_r, ky_r)
    ax_r.set_xlabel('kx'),ax_r.set_ylabel('ky')
    ax_a = fig.add_subplot(122, projection='3d')
    ax_a.scatter(kx_a,ky_a)
    ax_a.set_xlabel('kx'),ax_a.set_ylabel('ky')
    plt.savefig('data/k_r.png')
    plt.show()
    return cartesian_coords, cartesian_coords_abs
k_points, k_points_abs = convert_to_cartesian(k_points, a=1, a_abs=3.98E-9, c=1)
def cal_k_sym(k_points):
    #k_points_all = k_points_pro#
    # 这个只是复制了引用给他，后续在k_points_all.append时会动态改变K_points_pro的，越来越长，最后爆掉内存
    k_points_all = k_points[:] #这样才是赋值，创建新列表
    kx, ky = [], []
    #旋转变换
    for i in range(1,6):
        theta = np.pi * i / 3
        print(i)
        a=1
        for j, (kx0,ky0) in enumerate(k_points):
            kx_1 = kx0 * np.cos(theta) + ky0 * np.sin(theta)
            ky_1 = -kx0 * np.sin(theta) + ky0 * np.cos(theta)
            a+=1
            print(a)
            print((kx_1,ky_1))
            k_points_all.append((kx_1,ky_1))
    # 镜像对称
    k_points_all1 = k_points_all[:]
    # 这里也容易出一样的问题
    for i, (kx0,ky0) in enumerate(k_points_all1):
        ky_2 = -ky0
        k_points_all.append((kx0,ky_2))
    for i, (kx0,ky0) in enumerate(k_points_all):
        kx.append(kx0)
        ky.append(ky0)
    k_points_all2 = np.vstack([kx, ky]).T
    #np.savetxt('data/tunnelp_kxky_u.txt', k_points_all2)
    plt.scatter(kx,ky)
    plt.show()
    plt.clf()
    return k_points_all2

k_points_all = cal_k_sym(k_points)
kx = k_points_all[:,0]
ky = k_points_all[:,1]
# 2D Color-coded plot
fig, ax = plt.figure(figsize=(10, 8)), plt.axes()
scatter = ax.scatter(kx, ky, c=mass_eff_ratio_values_all, cmap='rainbow', norm=LogNorm())
plt.colorbar(scatter, label='Effective Mass Ratio (log scale)')
ax.set_xlabel('kx')
ax.set_ylabel('ky')
lim = max(abs(kx).max(), abs(ky).max())*1.1
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')  # Ensuring equal unit lengths for kx and ky axes
plt.title('2D Plot of Effective Mass Ratios (Color-coded, Rainbow Colormap, Matched Scales)')
plt.show()

# Define the vertices of the hexagon that bounds the first Brillouin zone
hexagon_vertices = np.array([
    [1, 0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1, 0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2],
    [1, 0]  # Closing the path by repeating the first vertex
])

from matplotlib.path import Path
hexagon_path = Path(hexagon_vertices, closed=True)

def process_brillouin_zone(kx, ky, mass_eff):
    # Initialize the full arrays with the original data
    kx_full = kx.copy()
    ky_full = ky.copy()
    mass_eff_full = mass_eff.copy()
    # Apply rotations and reflections to populate the Brillouin zone

    # Combine kx, ky, and mass_eff into a single array to handle them together
    combined = np.vstack((kx_full, ky_full, mass_eff_full)).T

    # Remove duplicate points by using a structured numpy array to enable easy checking of duplicates
    dtype = [('kx', 'f8'), ('ky', 'f8'), ('mass_eff', 'f8')]
    structured_array = np.array(list(map(tuple, combined)), dtype=dtype)
    unique_structured_array = np.unique(structured_array, axis=0)

    # Split the arrays back into kx, ky, and mass_eff
    kx_unique, ky_unique, mass_eff_unique = unique_structured_array['kx'], unique_structured_array['ky'], unique_structured_array['mass_eff']

    return kx_unique, ky_unique, mass_eff_unique
kx,ky,mass_eff =process_brillouin_zone(kx,ky,mass_eff_ratio_values_all)
#%%
###
fig, ax = plt.figure(figsize=(10, 8)), plt.axes()
scatter = ax.scatter(kx, ky, c=mass_eff, cmap='rainbow', norm=LogNorm())
plt.colorbar(scatter, label='Effective Mass Ratio (log scale)')
ax.set_xlabel('kx')
ax.set_ylabel('ky')
lim = max(abs(kx).max(), abs(ky).max())*1.1
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')  # Ensuring equal unit lengths for kx and ky axes
plt.title('2D Plot of Effective Mass Ratios (Color-coded, Rainbow Colormap, Matched Scales)')
plt.show()
#%%
# Create a grid for interpolation
grid_x, grid_y = np.mgrid[np.min(kx):np.max(kx):1000j, np.min(ky):np.max(ky):1000j]
from scipy.interpolate import griddata
# Perform interpolation over the full grid
grid_z = griddata((kx, ky), mass_eff, (grid_x, grid_y), method='linear')

# Mask out values outside the hexagon
# Create a mask for points outside the hexagon
inside_hexagon = hexagon_path.contains_points(np.vstack((grid_x.ravel(), grid_y.ravel())).T)
inside_hexagon = inside_hexagon.reshape(grid_x.shape)  # Reshape the mask back to the grid shape
grid_z[~inside_hexagon] = np.nan  # Set values outside the hexagon to NaN
#%%
# Plot the interpolated 2D map of effective mass
plt.figure(figsize=(10, 8))
plt.imshow(grid_z.T, extent=[np.min(grid_x), np.max(grid_x), np.min(grid_y), np.max(grid_y)], aspect='equal', origin='lower', cmap='rainbow',  norm=LogNorm())
plt.scatter(grid_x, grid_y, c=grid_z, cmap='rainbow', s=0.1,norm=LogNorm(),  label='Data Points')
plt.colorbar(label='Effective Mass Ratio')
plt.xlabel('kx')
plt.ylabel('ky')
plt.title('Continuous 2D Mapping of Effective Mass in the First Brillouin Zone')
plt.show()

#%%
fig, ax = plt.figure(figsize=(10, 8)), plt.axes()
scatter = ax.scatter(grid_x, grid_y, c=grid_z, cmap='rainbow', s=0.1,  label='Data Points')
cbar = plt.colorbar(scatter, label='Effective Mass Ratio (log scale)')
#cbar.set_ticks([0.1,1,10])
ax.set_xlabel('kx')
ax.set_ylabel('ky')
lim = max(abs(kx).max(), abs(ky).max())*1.1
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')  # Ensuring equal unit lengths for kx and ky axes
plt.title('2D Plot of Effective Mass Ratios (Color-coded, Rainbow Colormap, Matched Scales)')
plt.show()