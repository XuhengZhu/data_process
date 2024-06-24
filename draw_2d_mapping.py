import numpy as np
from matplotlib.colors import LogNorm
x_y_path = r"D:\Users\Desktop\计算\GaSe\kx ky.txt"
x, y = np.loadtxt(x_y_path, usecols=(0,1), unpack=True)

mass_eff_ratio_path = r"D:\Users\Desktop\计算\GaSe\band_line_energy\out\mass_eff_ratio.dat"
mass_eff_ratio = np.loadtxt(mass_eff_ratio_path,usecols=1, unpack=True)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
name_colobar = 'Effective Mass Ratio'
name_xlabel, name_ylabel, name_ylabel = 'kx', 'ky','Effective Mass'
name_title = '3D Plot of Effective Mass Ratio'
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, mass_eff_ratio, c= mass_eff_ratio, cmap='rainbow', norm = LogNorm())
plt.colorbar(scatter, label= name_colobar)
ax.set_xlabel(name_xlabel),ax.set_ylabel(name_ylabel)
ax.set_zlabel(name_colobar)
ax.set_aspect('auto')
plt.title(name_title)
plt.show()

# 2D color mapping
fig, ax = plt.figure(figsize=(10, 10)), plt.axes()
scatter = ax.scatter(x, y, c=mass_eff_ratio, cmap='rainbow', norm=LogNorm())
plt.colorbar(scatter, label='Effective Mass Ratio (log scale)')
ax.set_xlabel(name_xlabel),ax.set_ylabel(name_ylabel)
ax.set_aspect('equal')  # Equal unit lengths for kx and ky axes
lim = 0.55
ax.set_xlim(-0.02,lim)
ax.set_ylim(-0.020,lim)
plt.title('2D Plot of Effective Mass Ratios (Color-coded, Rainbow Colormap)')
plt.show()



def rotate_points(x,y,angle=60):
    theta = np.radians(angle)
    cos_ang, sin_ang = np.cos(theta), np.sin(theta)
    x_rot = cos_ang*x - sin_ang*y
    y_rot = sin_ang*x + cos_ang*y
    return x_rot,y_rot
def process_brillouin_zone(kx,ky,mass_eff_ratio):
    kx_full = kx.copy()
    ky_full = ky.copy()
    mass_eff_ratio_full = mass_eff_ratio.copy()
    # 旋转和对称变换
    for i in range(1,6):
        angle = 60*i
        kx_rot, ky_rot =rotate_points(kx,ky,angle)
        kx_full = np.concatenate([kx_full,kx_rot,kx_rot])
        ky_full = np.concatenate([ky_full,ky_rot,-ky_rot])
        mass_eff_ratio_full = np.concatenate([mass_eff_ratio_full,mass_eff_ratio,mass_eff_ratio])
    # 放到一起便于处理
    combined = np.vstack((kx_full,ky_full,mass_eff_ratio_full)).T
    # 删除变换导致的重复k点
    dtype = [('kx','f8'),('ky','f8'),('mass_eff_ratio','f8')]
    structured_array = np.array(list(map(tuple,combined)),dtype=dtype)
    unique_structured_array = np.unique(structured_array,axis = 0)

    # 将数据拆分为kx，ky，z
    kx_unique, ky_unique, mass_eff_ratio_unique = unique_structured_array['kx'], unique_structured_array['ky'],unique_structured_array['mass_eff_ratio']
    return kx_unique, ky_unique, mass_eff_ratio_unique
kx_unique, ky_unique, mass_eff_ratio_unique = process_brillouin_zone(x,y, mass_eff_ratio)
from scipy.interpolate import griddata

# Define the vertices of the hexagon that bounds the first Brillouin zone
hexagon_vertices = np.array([
    [1, 0],  # Example coordinates, adjust according to your specific Brillouin zone
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1, 0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2],
    [1, 0]  # Close the path by repeating the first point
])
hexagon_vertices = 0.5*hexagon_vertices
from matplotlib.path import Path
hexagon_path = Path(hexagon_vertices)

grid_x, grid_y = np.mgrid[np.min(kx_unique):np.max(kx_unique):1000j,np.min(ky_unique):np.max(ky_unique):1000j]


# Interpolate over the full grid
grid_z = griddata((kx_unique, ky_unique), mass_eff_ratio_unique, (grid_x, grid_y), method='cubic')
print(np.vstack((grid_x.ravel(), grid_y.ravel())).T)
# Apply the hexagon mask
inside_hexagon = hexagon_path.contains_points(np.vstack((grid_x.ravel(), grid_y.ravel())).T)
grid_z[~inside_hexagon] = np.nan  # Set points outside the hexagon to NaN
# plot
plt.figure(figsize=(10, 10))
plt.imshow(grid_z, extent=[np.min(grid_x), np.max(grid_x), np.min(grid_y), np.max(grid_y)], origin='lower', cmap='rainbow', aspect='equal')
plt.colorbar(label='Effective Mass Ratio')
plt.xlabel('kx')
plt.ylabel('ky')
plt.title('Continuous 2D Mapping of Effective Mass in the First Brillouin Zone')
plt.show()