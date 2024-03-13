import numpy as np
import matplotlib.pyplot as plt
import tmr_vdos
from scipy.integrate import simps


# 修正的函数来读取并解析文件，这次加入了对空行的检查以及对数据结构的调整
def read_eigenval_file_corrected(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # 跳过前七行基本信息
    lines = lines[7:]

    # 解析数据
    k_points = []
    bands = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 跳过空行
        if not line:
            i += 1
            continue

        # 读取K点信息
        k_info = line.split()
        if len(k_info) == 4:
            kx, ky, kz, weight = [float(x) for x in k_info]
            kx, ky = round(kx, 6), round(ky, 6)
            i += 1  # 移动到能带数据行
            band_info = []

            # 继续读取直到遇到下一个K点信息或文件结束
            while i < len(lines) and len(lines[i].split()) == 5:
                band_line = lines[i].strip().split()
                band_index, spin_up, spin_down, occ_up, occ_down = [
                    int(band_line[0]), float(band_line[1]), float(band_line[2]),
                    float(band_line[3]), float(band_line[4])
                ]
                band_info.append((band_index, spin_up, spin_down, occ_up, occ_down))
                i += 1

            k_points.append((kx, ky, kz, weight))
            bands.append(band_info)
        else:
            # 非预期行，跳过
            i += 1

    return k_points, bands


filepath = r'D:\Users\Desktop\计算\FeGaTe\414113\EIGENVAL'
k_points, bands = read_eigenval_file_corrected(filepath)


# 定义转换函数，将倒格矢坐标转换为笛卡尔坐标
def convert_to_cartesian(k_points, a=1,a_abs= 3.98E-9, c=1):
    # 倒格矢基矢在笛卡尔坐标系中的表示
    b1 = np.array([2 * np.pi / a, 2 * np.pi / (a * np.sqrt(3)), 0])
    b2 = np.array([0, 4 * np.pi / (a * np.sqrt(3)), 0])
    b1 = b1 / (2 * np.pi)
    b2 = b2 / (2 * np.pi)
    # 转换所有K点
    cartesian_coords = []
    cartesian_coords_abs = []
    for kx, ky, kz, weight in k_points:
        cartesian_k = kx * b1 + ky * b2  # 这里忽略kz因为我们关注于x-y平面
        cartesian_coords.append((cartesian_k[0], cartesian_k[1], kz, weight))
    cartesian_coords_abs =2 * np.pi* cartesian_coords/a_abs # 这个写的有点傻，后面在改

    return cartesian_coords, cartesian_coords_abs


# 执行转换
k_points,k_points_abs = convert_to_cartesian(k_points)


# 定义计算高斯波包的函数
def gaussian(x, mu, sigma):
    """计算高斯波包"""
    return 1.0 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def lorentzian(x, x0, sigma):
    return (1 / np.pi) * (sigma / ((x - x0) ** 2 + sigma ** 2))


# 定义计算K-resolved DOS的函数
def calculate_k_resolved_dos(k_points, bands, energy_range, delta_e, sigma):
    """
    计算K-resolved DOS。
    k_points: K点列表。
    bands: 对应于k_points的能带信息。
    energy_range: 能量范围 (min_energy, max_energy)。
    delta_e: 能量步长。
    sigma: 高斯波包宽度。
    """
    min_energy, max_energy = energy_range
    energies = np.arange(min_energy, max_energy, delta_e)
    dos_map_spin_up = np.zeros((len(k_points), len(energies)))
    dos_map_spin_up_lot = np.zeros((len(k_points), len(energies)))
    dos_map_spin_down = np.zeros((len(k_points), len(energies)))
    dos_map_spin_down_lot = np.zeros((len(k_points), len(energies)))
    for i, (_, _, _, _) in enumerate(k_points):
        for band in bands[i]:
            _, e_spin_up, e_spin_down, _, _ = band
            # 对spin up和spin down分别计算并累加贡献
            # print(e_spin_up,e_spin_down)
            dos_map_spin_up[i] += gaussian(energies, e_spin_up, sigma)
            dos_map_spin_down[i] += gaussian(energies, e_spin_down, sigma)
            # dos_map_spin_up_lot[i] += lorentzian(energies, e_spin_up, sigma)
            # dos_map_spin_down_lot[i] += lorentzian(energies, e_spin_down, sigma)
    return (energies, dos_map_spin_up, dos_map_spin_down, dos_map_spin_up_lot, dos_map_spin_down_lot)


# 设置参数
energy_range = (-10, 10)  # 示例能量范围，根据实际数据调整
delta_e = 0.002  # 能量步长
sigma = 0.01  # 高斯波包宽度

# 计算K-resolved DOS,将数据打个包免得慢慢写一堆变量
data = calculate_k_resolved_dos(k_points, bands, energy_range, delta_e, sigma)
data = (k_points,) + data  # 向元组中添加元素


# data = (k_points,energies, dos_map_spin_up, dos_map_spin_down, dos_map_spin_up_lot, dos_map_spin_down_lot)

def weight_kpoint(k_point,U,Ef,E=0,a=3.98E-9, d=6E-9): #给归一化的相对坐标
    eV = 1.6022E-19
    h_bar = 1.05257E-34
    m = 9.1E-31
    m_eff = 0.1
    kx, ky = k_point[0]/a, k_point[1]/a
    k = (np.sqrt(2*m*eV*(U-Ef+E)+(h_bar**2)*(kx**2 + ky**2)))/h_bar
    weight = np.exp(-2 * k * d)
    return weight

### 绘出二维平面内K点对应权重大小
def draw_weights(k_points):
    weight = []
    for k_point in k_points:
        weight.append(weight_kpoint(k_point, 1.0, 0, 0,a=3.98E-9,d=6E-9 ))


draw_weights(k_points)

def weight_kpoints(k_points,Us, Ef, Es=0):
    h_bar = 1.05257E-34
    m = 9.1E-31
    m_eff = 0.1
    weights = np.zeros(len(k_points))
    for i in len(k_points):
        kx, ky = k_points[i][0], k_points[i][1]
        weights[i] = np.exp(np.sqrt(Us[i]- Ef+ Es+ (h_bar**2)*(kx**2 + ky**2)/(2*m_eff*m)))
    return weights

def cal_tdos(data, E_fermi):
    k_points, energies, dos_map_spin_up, dos_map_spin_down, dos_map_spin_up_lot, dos_map_spin_down_lot = data[0], data[
        1], data[2], data[3], data[4], data[5]
    tdos_up = np.zeros(len(energies))
    tdos_down = np.zeros(len(energies))
    tdos_tmr_up = np.zeros(len(energies))
    tdos_tmr_down = np.zeros(len(energies))
    tdos_up_lot = np.zeros(len(energies))
    tdos_down_lot = np.zeros(len(energies))
    for i, (kx, ky, _, _) in enumerate(k_points):
        tdos_up += dos_map_spin_up[i] * k_points[i][3]
        # 所有K点权重为1，直接相加，检查高斯展宽计算方法与四面体方法是否一致，非相干隧穿的简单Julliere模型使用
        tdos_down += dos_map_spin_down[i] * k_points[i][3]
        # tdos_up_lot += dos_map_spin_up_lot[i]*k_points[i][3]
        # tdos_down_lot += dos_map_spin_down_lot[i] * k_points[i][3]
        # 计算单个K点加权的DOS，然后算TMR，还是先算加权的电导，然后再单独平行态和反平行态相加乘以权重？
        # 先计算电导，再将电导加权这样才正确

    energies = energies - E_fermi
    # tdos_up = np.convolve(tdos_up,np.ones(5))/5
    # tdos_down = np.convolve(tdos_down, np.ones(5))/5
    return energies, tdos_up, tdos_down


def cal_tmr_weight(data, E_fermi):
    k_points, energies, dos_map_spin_up, dos_map_spin_down, dos_map_spin_up_lot, dos_map_spin_down_lot = data[0], data[
        1], data[2], data[3], data[4], data[5]
    vb_range = np.linspace(-1.5, 1.5, 1000)
    for i, (_, _, _, _) in enumerate(k_points):
        conductance_p, conductance_ap = k_points[i][3]*tmr_vdos.cal_conduct(vb_range, dos_map_spin_up[i], dos_map_spin_down[i])

# 计算单gamma点dos及相关tmr
def cal_dos_k0(k_points, energies, dos_map_spin_up, dos_map_spin_down, E_fermi):
    dos_k0_up = dos_map_spin_up[0]
    dos_k0_down = dos_map_spin_down[0]
    return dos_k0_up, dos_k0_down


E_fermi = 6.638
energies, tdos_up, tdos_down = cal_tdos(data, E_fermi)

dos_k0_up, dos_k0_down = cal_dos_k0(data)

plt.figure(figsize=(5, 5))
plt.plot(energies, tdos_up, linewidth=1, label='tdos_up')
plt.plot(energies, 0 - tdos_down, linewidth=1, label='tdos_down')
plt.xlim(-3, 3), plt.ylim(-30, 30)
plt.legend()
plt.show()

# gamma 点
plt.figure(figsize=(5, 5))
plt.plot(energies, dos_k0_up, linewidth=1, label='tdos_up')
plt.plot(energies, 0 - dos_k0_down, linewidth=1, label='tdos_down')
plt.xlim(-3, 3), plt.ylim(-30, 30)
plt.legend()
plt.show()

data = np.vstack([energies, tdos_up, tdos_down]).T
# data_tmr_w =np.vstack([energies, tdos_tmr_up, tdos_tmr_down]).T
# 指定保存路径，更改为你的目标路径
save_path = r'D:\Users\Desktop\计算\FeGaTe\414113\a'

# 保存数据到txt文件
np.savetxt(save_path + 'tdos.dat', data, fmt='%f', header='Energy DOS_Up DOS_Down', comments='')

# 提取所有K点的Kx和Ky值
kx_values = [k[0] for k in k_points]
ky_values = [k[1] for k in k_points]

# 在平面上绘制所有（Kx，Ky）点的位置
plt.figure(figsize=(8, 6))
plt.scatter(kx_values, ky_values, color='blue', label='K points')
plt.xlabel('Kx')
plt.ylabel('Ky')
plt.title('Distribution of K points in the Kx-Ky plane')

plt.legend()
plt.grid(True)
plt.gca().set_aspect(1)
plt.show()

# 提取转换后的K点的Kx和Ky值进行绘制
cartesian_kx_values = [k[0] for k in k_points]
cartesian_ky_values = [k[1] for k in k_points]

# 在平面上绘制转换后的（Kx，Ky）点的位置
plt.figure(figsize=(8, 6))
plt.scatter(cartesian_kx_values, cartesian_ky_values, color='red', label='K points in Cartesian')
plt.xlabel('Kx (Cartesian)')
plt.ylabel('Ky (Cartesian)')
plt.title('Distribution of K points in Cartesian Coordinates')
plt.legend()
plt.grid(True)
plt.gca().set_aspect(1)
plt.show()

'''
# 保存转换后的K点坐标
cartesian_coords_filepath = '/mnt/data/cartesian_k_points.txt'
np.savetxt(cartesian_coords_filepath, np.array(cartesian_k_points), fmt='%f', header='Kx Ky Kz Weight')

cartesian_coords_filepath
'''
