import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tmr_vdos
from scipy.integrate import simps
import test5
import pickle
import pprint
# %%
# 修正的函数来读取并解析文件，这次加入了对空行的检查以及对数据结构的调整
def read_eigenval_file_corrected(filepath,e_fermi):
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
            kx = round(kx, 6)
            ky = round(ky, 6)
            i += 1  # 移动到能带数据行
            band_info = []

            # 继续读取直到遇到下一个K点信息或文件结束
            while i < len(lines) and len(lines[i].split()) == 5:
                band_line = lines[i].strip().split()
                band_index, spin_up, spin_down, occ_up, occ_down = [
                    int(band_line[0]), float(band_line[1]), float(band_line[2]),
                    float(band_line[3]), float(band_line[4])
                ]
                band_info.append((band_index, spin_up-e_fermi, spin_down-e_fermi, occ_up, occ_down))
                i += 1

            k_points.append((kx, ky, kz, weight))
            bands.append(band_info)
        else:
            # 非预期行，跳过
            i += 1

    return k_points, bands

# e_fermi = 6.638 #FeGaTe
e_fermi = 6.6376
filepath = r"D:\Users\Desktop\计算\FeGaTe\47 47 17\EIGENVAL"
k_points_xyz, bands = read_eigenval_file_corrected(filepath, e_fermi)


# %%
# 定义转换函数，将倒格矢坐标转换为笛卡尔坐标
def convert_to_cartesian(k_points, a=1, a_abs=3.98E-9, c=1):
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
        cartesian_coords_abs.append(
            (2 * np.pi * cartesian_k[0] / a_abs, 2 * np.pi * cartesian_k[1] / a_abs, kz, weight))
    #-r和-a分别为绝对坐标和相对坐标
    kx_r = [ki[0] for ki in cartesian_coords]
    kx_a = [ki[0] for ki in k_points]
    ky_r = [ki[1] for ki in cartesian_coords]
    ky_a = [ki[1] for ki in k_points]
    kz_r = [ki[2] for ki in cartesian_coords]
    kz_a = [ki[2] for ki in k_points]
    fig =plt.figure()
    ax_r = fig.add_subplot(121, projection='3d')
    ax_r.scatter(kx_r, ky_r, kz_r)
    ax_r.set_xlabel('kx'),ax_r.set_ylabel('ky'),ax_r.set_zlabel('kz')
    ax_a = fig.add_subplot(122, projection='3d')
    ax_a.scatter(kx_a,ky_a,kz_a)
    ax_a.set_xlabel('kx'),ax_a.set_ylabel('ky'),ax_a.set_zlabel('kz')
    plt.savefig('data/k_r.png')
    plt.show()
    return cartesian_coords, cartesian_coords_abs


# 执行转换
k_points_xyz, k_points_xyz_abs = convert_to_cartesian(k_points_xyz)
pprint.pprint(k_points_xyz)
#pprint.pprint(k_points_abs)
np.savetxt('data/kxkykzweight.txt', k_points_xyz)


# %%
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
    dos_map_spin_down = np.zeros((len(k_points), len(energies)))

    # 创建字典，以(kx,ky)为键，值为dos列表，存储投影dos信息
    k_points_pro = []
    dos_map_spin_up_pro = {}
    dos_map_spin_down_pro = {}
    dos_map_spin_up_pro_nosymweigh = {}
    dos_map_spin_down_pro_nosymweigh = {}
    for i, (kx, ky, _, weigh) in enumerate(k_points):
        for band in bands[i]:
            _, e_spin_up, e_spin_down, _, _ = band
            # 对spin up和spin down分别计算并累加贡献
            # print(e_spin_up,e_spin_down)
            # 此处为三维上每一个K点的DOS，这么做意义不大
            dos_map_spin_up[i] += gaussian(energies, e_spin_up, sigma)
            dos_map_spin_down[i] += gaussian(energies, e_spin_down, sigma)
        if (kx,ky) not in dos_map_spin_up_pro:
            # 这里则是用来做相干计算的投影到kxky平面内的DOS
            k_points_pro.append((kx,ky,weigh))
            dos_map_spin_up_pro[(kx,ky)] = np.zeros(len(energies))
            dos_map_spin_down_pro[(kx,ky)] = np.zeros(len(energies))
            dos_map_spin_up_pro_nosymweigh[(kx, ky)] = np.zeros(len(energies))
            dos_map_spin_down_pro_nosymweigh[(kx, ky)] = np.zeros(len(energies))
        dos_map_spin_up_pro[(kx,ky)] += dos_map_spin_up[i]*weigh # 有权重仅用于需要在整个平面求和计算总dos和电导用
        dos_map_spin_down_pro[(kx,ky)] += dos_map_spin_down[i]*weigh
        dos_map_spin_up_pro_nosymweigh[(kx, ky)] += dos_map_spin_up[i]  # 无权重仅用于画图用，后面不用乘以这个权重了对于prodos
        dos_map_spin_down_pro_nosymweigh[(kx, ky)] += dos_map_spin_down[i]
    # 将字典转换为列表方便处理
    #dos_map_spin_up_down
    return (energies, dos_map_spin_up, dos_map_spin_down), (energies, dos_map_spin_up_pro, dos_map_spin_down_pro),k_points_pro,(energies, dos_map_spin_up_pro_nosymweigh, dos_map_spin_down_pro_nosymweigh)



# 设置参数
h_bar = 1.05257E-34
energy_range = (-3, 3)  # 示例能量范围，根据实际数据调整
delta_e = 0.002  # 能量步长
sigma = 0.028  # 高斯波包宽度
#E_fermi = 6.638
d=5E-9
m = 9.1E-31
m_eff = 0.1
epsilon = 1E-10 # 一个小常数，防止除以0
# 计算K-resolved DOS,将数据打个包免得慢慢写一堆变量

data_k_xyz_dos, data_k_xy_dos_pro, k_points_xy_pro,data_k_xy_pro_noweigh = calculate_k_resolved_dos(k_points_xyz, bands, energy_range, delta_e, sigma)
# 存一下数据
with open('data/data_pro474713.pkl','wb') as f:
    pickle.dump(data_k_xy_dos_pro, f)
with open('data/data_pro_nosymweigh474713','wb') as g:
    pickle.dump(data_k_xy_pro_noweigh, g)


np.savetxt('data/k_points_xy_pro_474713.dat', k_points_xy_pro)

data_k_xyz_dos = (k_points_xyz,) + data_k_xyz_dos  # 向元组中添加元素

#%% 对称变换函数，补全k点
def cal_k_sym1(k_points_pro):
    k_points_all = k_points_pro.copy()

    kx = k_points_all[:,0]
    ky = k_points_all[:,1]
    weigh_sym = k_points_all[:,2]
    for i in range(1, 6):
        theta = np.pi * i / 3
        kx_1 = kx * np.cos(theta) + ky * np.sin(theta)
        ky_1 = -kx * np.sin(theta) + ky * np.cos(theta)
        k_points_all=np.append(k_points_all,np.vstack([kx_1, ky_1,weigh_sym]).T,axis=0)
    kx__ky =np.vstack([k_points_all[:,0],-k_points_all[:,1],k_points_all[:,2]]).T
    k_points_all = np.append(k_points_all,kx__ky,axis=0)
    np.savetxt('data/kxky_all_4747.dat', k_points_all)
    plt.scatter(k_points_all[:,0], k_points_all[:,1])
    plt.show()
    plt.clf()
    plt.close()
    fig, ax = plt.figure(figsize=(10, 8)), plt.axes()
    scatter = ax.scatter(k_points_all[:,0], k_points_all[:,1], c=k_points_all[:,2], cmap='rainbow')
    plt.colorbar(scatter, label='Ef')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    lim = max(abs(kx).max(), abs(ky).max()) * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')  # Ensuring equal unit lengths for kx and ky axes
    plt.title('2D Plot of Effective Mass Ratios (Color-coded, Rainbow Colormap, Matched Scales)')
    plt.show()
    return k_points_all

k_points_xy_all = cal_k_sym1(np.loadtxt('data/k_points_xy_pro_474713.dat'))

#%% 读取导带最低点信息文件
with open('data/474713cbm_map.pkl','rb') as f:
    cbm_map = pickle.load(f)
with open('data/474713vbm_map.pkl','rb') as d:
    vbm_map = pickle.load(d)
pprint.pprint(cbm_map)


# %%
# data = (k_points,energies, dos_map_spin_up, dos_map_spin_down)
# def weight_kpoint(k_point, U, Ef,m_eff, E=0, a=3.98E-10, d=5E-9):  # 给归一化的相对坐标
#     eV = 1.6022E-19
#     h_bar = 1.05257E-34
#     m = 9.1E-31
#     m_eff1 = m_eff
#     U1 = U
#     kx, ky = k_point[0] / a, k_point[1] / a
#     kapa = (np.sqrt(2 * m * m_eff1 * eV * (U1 - Ef + E) + (h_bar ** 2) * (kx ** 2 + ky ** 2))) / h_bar
#     kappa_00 = (np.sqrt(2 * m * m_eff1 * eV * (1.00 - Ef + E))) / h_bar #gamma点
#     weight_00 = np.exp(-2 * kappa_00 * d)
#     weight = np.exp(-2 * kapa * d) / weight_00
#     return weight
def weight_kpoint(k_point, U, Ef,m_eff, E=0, a=3.98E-10, d=5E-9):  # 给归一化的相对坐标
    eV = 1.6022E-19
    h_bar = 1.05257E-34
    m = 9.1E-31
    m_eff1 = m_eff
    U1 = U
    # 汪林望老师认为不需要加上平行方向的能量
    kapa = (np.sqrt(2 * m * m_eff1 * eV * (U1 - Ef + E))) / h_bar
    kappa_00 = (np.sqrt(2 * m * m_eff1 * eV * (1.00 - Ef + E))) / h_bar #gamma点
    weight_00 = np.exp(-2 * kappa_00 * d)
    weight = np.exp(-2 * kapa * d) / weight_00
    return weight


### 绘出二维平面内K点对应权重大小
def draw_weights(k_points_pro, cbm_map,vbm_map):
    weight_cbm, weight_vbm = [], []
    cbm_mapa, vbm_mapa = cbm_map, vbm_map
    kx, ky, weight_sym = [], [], []
    weight_k_only = [] # 只考虑K平行带来的隧穿，即不同k点的势垒高度认为相同
    # weight_sym 计算完投影的时候不需要，在计算投影的时候用就行
    for k_point in k_points_pro:
        U_cbm, U_vbm = cbm_mapa[(k_point[0], k_point[1])], vbm_mapa[(k_point[0]),k_point[1]]
        weight_cbm.append(weight_kpoint(k_point, U_cbm-1.0326, 0, m_eff, 0, a=3.98E-10, d=5E-9))
        weight_vbm.append(weight_kpoint(k_point, 2.0326-U_vbm, 0, m_eff, 0, a=3.98E-10, d=5E-9))
        #weight_k_only没什么用了
        weight_k_only.append(weight_kpoint(k_point,1.00,0, m_eff, 0, a=3.98E-10, d=5E-9))
        kx.append(k_point[0])
        ky.append(k_point[1])

    kxky_weight_cbm = np.vstack([kx, ky, weight_cbm]).T
    kxky_weight_vbm = np.vstack([kx,ky,weight_vbm]).T
    tunnelp_k = np.vstack([kx,ky,weight_k_only]).T
    np.savetxt('data/kxkyweight_cbm.txt', kxky_weight_cbm)
    np.savetxt('data/kxkyweight_vbm.txt',kxky_weight_vbm)
    np.savetxt('data/tunnelp_kxky_only.txt', tunnelp_k)
    # pprint.pprint(np.vstack([kx, ky, weight_cbm]).T)

    weight_k = weight_k_only[:]
    weight_cbm_all = weight_cbm[:]
    weight_vbm_all = weight_vbm[:]
    #weight_k_only_all = weight_k_only[:]
    for i in range(1,12):
        # weight_all.append(weight), 这样是将列表作为一个元素添加到后面，错了
        weight_cbm_all.extend(weight_cbm)
        weight_vbm_all.extend(weight_vbm)
    kxky = np.loadtxt('data/kxky_all_4747.dat')
    fig, ax = plt.figure(figsize=(10, 8)), plt.axes()
    from matplotlib.colors import LogNorm
    scatter = ax.scatter(np.array(kxky[:,0]), np.array(kxky[:,1]), c=weight_vbm_all, cmap='rainbow', norm=LogNorm())
    plt.colorbar(scatter, label='')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    lim = max(abs(kxky[:,0]).max(), abs(kxky[:,1]).max()) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')  # Ensuring equal unit lengths for kx and ky axes
    plt.title('2D Plot of weight')
    plt.show()
    return weight_cbm, weight_vbm, weight_k

weight_cbm, weight_vbm, weight_k = draw_weights(k_points_xy_pro, cbm_map, vbm_map)
#%%
pprint.pprint(weight_vbm)
# %%
#或许可以加快点效率
def weight_kpoints(k_points, Us, Ef,m_eff, Es=0):
    h_bar = 1.05257E-34
    m = 9.1E-31
    m_eff = m_eff
    weights = np.zeros(len(k_points))
    for i in len(k_points):
        kx, ky = k_points[i][0], k_points[i][1]
        weights[i] = np.exp(np.sqrt(Us[i] - Ef + Es + (h_bar ** 2) * (kx ** 2 + ky ** 2) / (2 * m_eff * m)))
    return weights


#%%
import test5
def draw_p_dos_data(data_pro, k_points_xy_pro,e): # e是能量，是画能量为e处对应的投影态密度
    kxky= np.asarray(k_points_xy_pro)
    kx = kxky[:,0]
    ky = kxky[:,1]
    energies = data_pro[0]
    e_close = np.isclose(energies, e, atol=1e-8)  # atol是容忍度
    index_e = np.where(e_close)[0][0]
    print(index_e)
    dos_map_spin_up = data_pro[1]
    dos_map_spin_down = data_pro[2]
    dos_down_e = np.zeros(len(kx))
    dos_up_e = np.zeros(len(kx))
    print(len(kx))
    for i in range(len(kx)):
        dos_up_e[i] = dos_map_spin_up[(kx[i],ky[i])][index_e]
        dos_down_e[i] = dos_map_spin_down[(kx[i],ky[i])][index_e]
    dos_up_e_all = dos_up_e[:]
    dos_down_e_all = dos_down_e[:]
    for i in range(1,12):
        # weight_all.append(weight), 这样是将列表作为一个元素添加到后面，错了
        dos_up_e_all= np.append(dos_up_e_all,dos_up_e)
        dos_down_e_all=np.append(dos_down_e_all,dos_down_e)
        print(i)
    kx_all = k_points_xy_all[:, 0]
    ky_all = k_points_xy_all[:, 1]
    #删除重复数据
    #atol = 1e-7
    #用pandas dataframe处理
    #print(isinstance(kx,np.ndarray),kx.ndim)
    df = pd.DataFrame({
        'kx':kx_all,
        'ky':ky_all,
        'dos_up_e_all':dos_up_e_all,
        'dos_down_e_all':dos_down_e_all
    })
    df[['kx','ky']]=df[['kx','ky']].round(8)
    #去除重复值并计算平均
    df_unique = df.drop_duplicates(subset=['kx','ky']).groupby(['kx','ky']).mean().reset_index()
    kx_unique = df_unique['kx'].to_numpy()
    ky_unique = df_unique['ky'].to_numpy()
    dos_up_e_unique = df_unique['dos_up_e_all'].to_numpy()
    dos_down_e_unique = df_unique['dos_down_e_all'].to_numpy()
    # Split the arrays back into kx, ky, and mass_eff

    print(len(kx_unique), len(kx_all))

    kx1,ky1,dos_up_matr = test5.smooth_data(kx_unique,ky_unique,dos_up_e_unique,0.75)
    kx1, ky1, dos_down_matr = test5.smooth_data(kx_unique, ky_unique, dos_down_e_unique, 0.75)
    return(kx1,ky1,dos_up_matr,dos_down_matr)

import pandas as pd
with open('data/data_pro_nosymweigh474713','rb') as f:
    loaded_data = pickle.load(f)
kxkya = np.loadtxt('data/k_points_xy_pro_474713.dat')
for e in (-0.5,-0.25,0,0.25,0.5):
    print(e)
    kx1,ky1,dos_up_matr,dos_down_matr = draw_p_dos_data(loaded_data,kxkya,e)

    fig1,ax1 = plt.subplots(figsize=(8,6))
    fig2,ax2 = plt.subplots(figsize=(8,6))
    im1 = ax1.imshow(dos_up_matr, extent=(-1, 1, -1, 1), cmap='viridis', interpolation='bilinear', origin='lower',
                     vmin=0, vmax=400)
    im2 = ax2.imshow(dos_down_matr, extent=(-0.7, 0.7, -0.7, 0.7), cmap='viridis', interpolation='bilinear',
                     origin='lower', vmin=0, vmax=400)

    cbar1,cbar2 = fig1.colorbar(im1, ax=ax1, orientation='vertical'),fig2.colorbar(im2, ax=ax2, orientation='vertical')
    ax1.set_aspect('equal'),ax2.set_aspect('equal')
    ax1.set_xticks([]),ax1.set_yticks([])
    ax2.set_xticks([]),ax2.set_yticks([])
    ax1.set_title(str(e)+'up')
    ax2.set_title(str(e)+'dn')
    fig1.savefig('fig/dos_FGaT_'+str(e)+'_up'+'.jpg'),fig2.savefig('fig/dos_FGaT_'+str(e)+'_down'+'.jpg')
    fig1.show()
    plt.close(fig1)
    fig2.show()
    plt.close(fig2)


#%% 求一圈一圈加起来的dos linear_k_dos
def cal_circle_sum_dos(data_k_xy_dos_pro, k_points_xy_pro,interval):
    energies, dos_map_up, dos_map_down = data_k_xy_dos_pro
    kii = []
    linear_k_dos_up,linear_k_dos_down = [],[]
    #for i, (kx, ky, _) in k_points_xy_pro:
    # enumerate是内置的迭代函数， enumerate()回返回枚举对象，每次循环时，i接收当前元素的索引，(kx,ky，_)接收当前元素的值
    for i, (kx, ky, _) in enumerate(k_points_xy_pro):
        ki = int(np.sqrt(kx**2+ky**2)//interval)
        if ki not in kii:
            kii.append(ki)
            print(ki)
            linear_k_dos_up.append(dos_map_up[(kx,ky)])
            linear_k_dos_down.append(dos_map_down[(kx,ky)])

        else:
            linear_k_dos_up[ki] = linear_k_dos_up[ki]+dos_map_up[(kx,ky)]
            linear_k_dos_down[ki] = linear_k_dos_down[ki]+dos_map_down[kx,ky]
    linear_k_dos_up, linear_k_dos_down = np.array(linear_k_dos_up), np.array(linear_k_dos_down)
    linear_k_dos = np.stack((linear_k_dos_up,linear_k_dos_down),axis=0)
    return linear_k_dos
linear_k_dos = cal_circle_sum_dos(data_k_xy_dos_pro, k_points_xy_pro,interval=0.03)

#%%
a = np.linspace(-1,1,5)
print(a)
#%%
def cal_linear_k_weight(vbm_map):
    return 0
def cal_circle_sum_dos_tmr(linear_k_dos,linear_k_weight):
    linear_k_dos_up =linear_k_dos[0,:,:]
    linear_k_dos_down = linear_k_dos[1,:,:]
    vb_range = np.linspace(-1.5, 1.5, 900)
    energies = np.linspace (-3,2.998,3000)
    # 对二维数组的列或者行遍历
    for i in range(linear_k_dos_up.shape[0]):
        conductance_p1, conductance_ap1 = tmr_vdos.cal_conduct_acc(vb_range, energies,linear_k_dos_up[i,:],linear_k_dos_down[i,:],1,0,5E-9 )
        #conductance_p1, conductance_ap1 = tmr_vdos.cal_conduct(vb_range, energies, linear_k_dos_up[i, :],linear_k_dos_down[i, :])
        epsilon = 1e-10
        if i == 0:
            conductance_p, conductance_ap = np.zeros_like(conductance_p1), np.zeros_like(conductance_p1)
        if i<5:
            conductance_p = conductance_p + conductance_p1
            conductance_ap = conductance_ap + conductance_ap1
    tmr = np.where(conductance_ap != 0, (conductance_p - conductance_ap) / (conductance_ap + epsilon), 0)
    #plt.figure(10,8)
    tmr_data = np.vstack((vb_range, tmr)).T
    np.savetxt('data/tmr_dataa.txt', tmr_data)
    plt.plot(vb_range,tmr)
    plt.show()
    return 0
a = cal_circle_sum_dos_tmr(linear_k_dos,0)


# %%
def cal_tdos(data, E_fermi):
    k_points, energies, dos_map_spin_up, dos_map_spin_down, = data[0], data[
        1], data[2], data[3]
    tdos_up = np.zeros(len(energies))
    tdos_down = np.zeros(len(energies))
    tdos_tmr_up = np.zeros(len(energies))
    tdos_tmr_down = np.zeros(len(energies))
    for i, (kx, ky, _, _) in enumerate(k_points):
        tdos_up += dos_map_spin_up[i] * k_points[i][3]
        # 所有K点权重为1，直接相加，检查高斯展宽计算方法与四面体方法是否一致，非相干隧穿的简单Julliere模型使用
        tdos_down += dos_map_spin_down[i] * k_points[i][3]
    vb_range = np.linspace(-1.5, 1.5, 500)
    conductance_p, conductance_ap = tmr_vdos.cal_conduct(vb_range, energies, tdos_up, tdos_down)
    tmr = np.where(conductance_ap != 0, (conductance_p - conductance_ap) / (conductance_ap+epsilon), 0)
    plt.figure(figsize=(14, 6))
    plt.subplot(121)

    plt.plot(vb_range, conductance_p, label='p')
    plt.plot(vb_range, conductance_ap, label='ap')
    plt.legend()
   #plt.show()
    # 绘制图表
    plt.subplot(122)
    plt.plot(vb_range, tmr, label='TMR vs Vb')
    plt.xlabel('Bias Voltage (V)')
    plt.ylabel('TMR')
    plt.title('TMR from extended Jullier model')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('fig/Jullier model.png', bbox_inches='tight',transparent = True)
    plt.clf()
    tdos = np.vstack((energies,tdos_up, tdos_down)).T
    np.savetxt('data/tdos_gause.txt', tdos)
    return energies, tdos_up, tdos_down
epsilon = 1E-10
cal_tdos(data_k_xyz_dos, 0)

#%% 计算01画圈的粗糙权重
def cal_dos_rough(data,E_fermi):
    k_points, energies, dos_map_spin_up, dos_map_spin_down, = data[0], data[
        1], data[2], data[3]
    tdos_up = np.zeros(len(energies))
    tdos_down = np.zeros(len(energies))

    for i, (kx, ky, _, _) in enumerate(k_points):
        if kx**2+ky**2 <=0.165:
            tdos_up += dos_map_spin_up[i] * k_points[i][3]
            # 所有K点权重为1，直接相加
            tdos_down += dos_map_spin_down[i] * k_points[i][3]
    dos_eff = np.transpose(np.vstack((energies,tdos_up, tdos_down)))
    path_a = r"D:\Users\Desktop\计算\FeGaTe\47 47 17"
    np.savetxt(path_a + '\dos_eff_FGaT.txt',dos_eff)
    vb_range = np.linspace(-1.5, 1.5, 500)
    conductance_p, conductance_ap = tmr_vdos.cal_conduct(vb_range, energies, tdos_up, tdos_down)
    tmr = np.where(conductance_ap != 0, (conductance_p - conductance_ap) / (conductance_ap + epsilon), 0)

    tmr_quan = np.transpose(np.vstack((vb_range,tmr)))
    np.savetxt(path_a + r'\tmr_huaquan.txt',tmr_quan)
    plt.figure(figsize=(14, 6))
    plt.subplot(121)

    plt.plot(vb_range, conductance_p, label='p')
    plt.plot(vb_range, conductance_ap, label='ap')
    plt.legend()
    # plt.show()
    # 绘制图表
    plt.subplot(122)
    plt.plot(vb_range, tmr, label='TMR vs Vb')
    plt.xlabel('Bias Voltage (V)')
    plt.ylabel('TMR')
    plt.title('TMR from huaquan')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('fig/画圈dos.png', bbox_inches='tight', transparent=True)
    plt.clf()
    return 0
cal_dos_rough(data_k_xyz_dos, 0)

#%% 点对点，处理计算 tmr方案，这个不对，不要用
# def cal_tmr_pp_weight(data, E_fermi):
#     k_points, energies, dos_map_spin_up, dos_map_spin_down = data[0], data[
#         1], data[2], data[3]
#     vb_range = np.linspace(-1.5, 1.5, 500)
#     conductance_p_total, conductance_ap_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
#     conductance_p_k_total, conductance_ap_k_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
#     # 先计算电导，再将电导加权这样才正确
#     for i, (_, _, _, _) in enumerate(k_points):
#         conductance_p, conductance_ap = tmr_vdos.cal_conduct(vb_range,energies, dos_map_spin_up[i],
#                                                                               dos_map_spin_down[i])
#         conductance_p_ku = k_points[i][3] * weight_ku_pp[i] *conductance_p
#         conductance_ap_ku = k_points[i][3] * weight_ku_pp[i] *conductance_ap
#         conductance_p_total += conductance_p_ku
#         conductance_ap_total += conductance_ap_ku
#
#         conductance_p_k = k_points[i][3] * weight_k_pp[i] * conductance_p
#         conductance_ap_k = k_points[i][3] * weight_k_pp[i] * conductance_ap
#         conductance_p_k_total += conductance_p_k
#         conductance_ap_k_total += conductance_ap_k
#     tmr = np.where(conductance_ap_total != 0, (conductance_p_total - conductance_ap_total) / (conductance_ap_total+epsilon), 0)
#     tmr_k = np.where(conductance_ap_k_total != 0,(conductance_p_k_total - conductance_ap_k_total)/(conductance_ap_k_total+epsilon), 0)
#
#
#     '''
#     plt.plot(vb_range,conductance_p_total, label='p')
#     plt.plot(vb_range,conductance_ap_total, label='ap')
#     plt.legend()
#     plt.show()
#     # 绘制图表
#     plt.figure(figsize=(8, 6))
#     '''
#     plt.figure(figsize=(14, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(vb_range, tmr, label='TMR vs Vb')
#     plt.xlabel('Bias Voltage (V)')
#     plt.ylabel('TMR')
#     plt.title('TMR pp k+u')
#     plt.grid(True)
#     plt.legend()
#
#     plt.subplot(1,2,2)
#     plt.plot(vb_range, conductance_p_k_total, label='p')
#     plt.plot(vb_range, conductance_ap_k_total, label='ap')
#     plt.legend()
#     plt.show()
#     # 绘制图表
#     plt.figure(figsize=(8, 6))
#     plt.plot(vb_range, tmr_k, label='TMR vs Vb')
#     plt.xlabel('Bias Voltage (V)')
#     plt.ylabel('TMR')
#     plt.title('TMR pp k-only')
#     plt.grid(True)
#     plt.legend()
#     plt.show()
#     plt.clf()
# cal_tmr_pp_weight(data,e_fermi)

#%%

mass_eff_ratio_path = r"D:\Users\Desktop\计算\GaSe\band_line_energy\out\mass_eff_ratio.dat"
mass_eff_ratio = np.loadtxt(mass_eff_ratio_path,usecols=1, unpack=True)



# %% 投影到kxky平面内处理
def cal_conduct_channel(data_pro):
    vb_range = np.linspace(-1.5, 1.5, 1500)
    # conductance_p_pro_total, conductance_ap_pro_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
    # conductance_p_pro_k_total, conductance_ap_pro_k_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
    energies, dos_up_pro, dos_down_pro =  data_pro[0], data_pro[1], data_pro[2]
    cudp_ctn_all = []
    cudap_ctn_all = []
    current_p_distr= []
    current_ap_distr= []
    kxky = []
    mass_n = 0
    for (kx,ky) in dos_down_pro:
        mass_n = mass_n + 1
        #conductance_p_pro, conductance_ap_pro = tmr_vdos.cal_conduct(vb_range, energies, dos_up_pro[(kx,ky)], dos_down_pro[(kx,ky)])
        conductance_p_pro, conductance_ap_pro = tmr_vdos.cal_conduct_acc(vb_range, energies, dos_up_pro[(kx, ky)],
                                                                     dos_down_pro[(kx, ky)],1,0,5e-9)
        # 电导分布
        # 连续电导分布 cudp_ctn
        cudp_ctn_all.append(conductance_p_pro)
        cudap_ctn_all.append(conductance_ap_pro)
        cudp,cudap = [],[]
        for i in (752,850,950,1050,1150,1250):
            cudp.append(conductance_p_pro[i])
            cudap.append(conductance_ap_pro[i])
        current_p_distr.append(cudp)
        current_ap_distr.append(cudap)

    cudp_ctn_all = np.array(cudp_ctn_all).T
    cudap_ctn_all = np.array(cudap_ctn_all).T
    np.savetxt('data/cudp_ctn_all.txt',cudp_ctn_all)
    np.savetxt('data/cudap_ctn_all.txt',cudap_ctn_all)
    #绘制电流分布 cudp1 为平行态，第一列按照k点排序的0v附近的导电通道，第2，3，4，5列分别为0.2 0.4 0.6 0.8 0.9v时的导电通道数目
    cudp1 = np.array(current_p_distr)
    cudap1 = np.array(current_ap_distr)
    # 可以分别存储导电通道和权重值在k空间的分布，这样可以方便地求总的电导
    np.savetxt( 'data/conduct_p_0246810.dat',cudp1)
    np.savetxt('data/conduct_ap_0246810.dat',cudap1)

    cudp2 = np.tile(cudp1,(12,1))
    cudap2 = np.tile(cudap1,(12,1))

    kxall = k_points_xy_all[:, 0]
    kyall = k_points_xy_all[:, 1]
    fig,axs = plt.subplots(2,6,figsize=(20,10))
    for i in range(0,6):
        cup,cuap = cudp2[:,i],cudap2[:,i]
        kx1, ky1, current_p_ma = test5.smooth_data(kxall, kyall, cup, 0.75)
        kx1, ky1, current_ap_ma = test5.smooth_data(kxall, kyall,cuap, 0.75)
        k_range=0.75
        ax1 = axs[0,i]
        ax2 = axs[1,i]
        ax1.imshow(current_p_ma, extent=(-1 * k_range, k_range, -1 * k_range, k_range), cmap='viridis',
                       interpolation='bilinear', origin='lower')
        ax2.imshow(current_ap_ma, extent=(-1 * k_range, k_range, -1 * k_range, k_range), cmap='viridis',
                       interpolation='bilinear', origin='lower')
        ax1.set_title('p')
        ax1.set_xticks([]),ax1.set_yticks([])
        ax2.set_title('ap')
        ax2.set_xticks([]),ax2.set_yticks([])
    plt.savefig('fig/conduc_channel_noweight474713.jpg')
    plt.show()
    plt.clf()


    return cudp1, cudap1, cudp_ctn_all, cudap_ctn_all

#这里计算不应该加乘上对称性的dos，对称权重只是在计算总的能量时用的
cudp1, cudap1, cudp_ctn_all, cudap_ctn_all = cal_conduct_channel(loaded_data)



#%% 将导电通道加权
def cal_conduct_channel_weight_tmr(cudp_ctn_all,cudap_ctn_all,weight):
    cudp_all = np.sum(cudp_ctn_all*weight, axis=1) # numpy会自动将weight进行广播以匹配形状 加总应该加上对称性的权重
    cudap_all = np.sum(cudap_ctn_all*weight, axis=1) # 可以沿着列轴相加
    tmr = (cudp_all-cudap_all)/cudap_all
    vb_range = np.linspace(-1.5, 1.5, 1500)
    # 开始画图
    plt.subplot(121)
    plt.plot(vb_range,tmr,color = 'r', label = 'tmr')
    plt.subplot(122)
    plt.plot(vb_range,cudp_all,color = 'g', label = 'current')
    plt.plot(vb_range,cudap_all,color = 'blue', label = 'cudap1')
    plt.show()
weight_a = np.ones((1,208))
#cal_conduct_channel_weight_tmr(cudp_ctn_all,cudap_ctn_all,weight_a)
#cal_conduct_channel_weight_tmr(cudp_ctn_all,cudap_ctn_all,weight_cbm)
cal_conduct_channel_weight_tmr(cudp_ctn_all,cudap_ctn_all,weight_vbm)
#%%
def cal_tmr_pro_weight_acc(data_pro, E_fermi): # 梯形势垒
    vb_range = np.linspace(-1.5, 1.5, 1500)
    conductance_p_pro_total, conductance_ap_pro_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
    conductance_p_pro_k_total, conductance_ap_pro_k_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
    energies, dos_up_pro, dos_down_pro =  data_pro[0], data_pro[1], data_pro[2]
    current_p_distr= []
    current_ap_distr= []
    kxky = []
    for (kx,ky) in dos_down_pro:
        print((kx,ky))
        U = cbm_map[(kx, ky)]
        kx,ky = (kx,ky)[0], (kx,ky)[1]
        e_parallel = (h_bar ** 2) * (kx ** 2 + ky ** 2)/(2*m)
        print(U)
        conductance_p_pro, conductance_ap_pro = tmr_vdos.cal_conduct_acc(vb_range, energies, dos_up_pro[(kx,ky)], dos_down_pro[(kx,ky)],1.0,e_parallel,d)
        # 电导分布
        cudp,cudap = [],[]

        for i in (752,800,900,1000,1100,1200):
            cudp.append(conductance_p_pro[i])
            cudap.append(conductance_ap_pro[i])
        print(cudp)
        current_p_distr.append(cudp)

        current_ap_distr.append(cudap)
        kxky.append([kx,ky])


        #conductance_p_pro_ku = conductance_p_pro * weight_kpoint((kx,ky,0,0),U-1.03 , 0,m_eff, 0, a=3.98E-10, d=5E-9)
        #conductance_ap_pro_ku = conductance_ap_pro * weight_kpoint((kx, ky, 0, 0), U-1.03, 0, m_eff, 0, a=3.98E-10, d=5E-9)
        conductance_p_pro_total += conductance_p_pro
        conductance_ap_pro_total += conductance_ap_pro

        conductance_p_pro_k = conductance_p_pro * weight_kpoint((kx,ky,0),1,0,m_eff,0,a=3.98E-10, d=5E-9)
        conductance_ap_pro_k = conductance_ap_pro * weight_kpoint((kx,ky,0),1,0,m_eff,0,a=3.98E-10, d=5E-9)
        #conductance_p_pro_k_total +=conductance_p_pro_k
        #conductance_ap_pro_k_total += conductance_ap_pro_k
    tmr = np.where(conductance_ap_pro_total !=0, (conductance_p_pro_total-conductance_ap_pro_total)/(conductance_ap_pro_total+epsilon), 0)
    #tmr = np.where(conductance_ap_pro_total != 0,(conductance_p_pro_total - conductance_ap_pro_total) / (conductance_ap_pro_total), 0)
    #tmr_k = np.where(conductance_ap_pro_k_total !=0,(conductance_p_pro_k_total-conductance_ap_pro_k_total)/(conductance_ap_pro_k_total+epsilon),0)
    '''
    plt.figure(figsize = (8, 6))
    plt.plot(vb_range, conductance_p_pro_total, label='p')
    plt.plot(vb_range, conductance_ap_pro_total, label='ap')
    plt.legend()
    plt.show()
    plt.close()
    '''
    #存储电导通道的分布数据为conduct_bias_pstate_kxky
    #np.savetxt(np.array(current_p_distr),'data/conduct_'+)

    #绘制电流分布
    cudp1 = np.array(current_p_distr)
    cudap1 = np.array(current_ap_distr)
    cudp2 = np.tile(cudp1,(12,1))
    cudap2 = np.tile(cudap1,(12,1))

    kxky = np.array(kxky)
    kxall = k_points_xy_all[:, 0]
    kyall = k_points_xy_all[:, 1]
    fig,axs = plt.subplots(2,6,figsize=(40,200))
    for i in range(0,6):
        cup,cuap = cudp2[:,i],cudap2[:,i]
        kx1, ky1, current_p_ma = test5.smooth_data(kxall, kyall, cup, 0.75)
        kx1, ky1, current_ap_ma = test5.smooth_data(kxall, kyall,cuap, 0.75)
        k_range=0.75
        ax1 = axs[0,i]
        ax2 = axs[1,i]
        im1 = ax1.imshow(current_p_ma, extent=(-1 * k_range, k_range, -1 * k_range, k_range), cmap='viridis',
                       interpolation='bilinear', origin='lower')
        im2 = ax2.imshow(current_ap_ma, extent=(-1 * k_range, k_range, -1 * k_range, k_range), cmap='viridis',
                       interpolation='bilinear', origin='lower')
        ax1.set_title('p')
        ax2.set_title('ap')
    plt.show(block=True)
    plt.clf()



    # 绘制tmr
    plt.figure(figsize = (8,6))
    plt.subplot(111)
    plt.plot(vb_range, tmr, label='TMR vs Vb')
    plt.xlabel('Bias Voltage (V)')
    plt.ylabel('TMR')
    plt.title('TMR pro k+u')
    plt.grid(True)

    plt.subplot(122)
    plt.plot(vb_range,tmr_k,label='TMR vs')
    plt.xlabel('Bias Voltage (V)')
    plt.ylabel('TMR')
    plt.title('TMR pro k-only')
    plt.legend()

    plt.show()
cal_tmr_pro_weight_acc(data_k_xy_dos_pro, 0)

# 计算单gamma点dos及相关tmr
def cal_dos_k0(k_points, energies, dos_map_spin_up, dos_map_spin_down, E_fermi):
    dos_k0_up = dos_map_spin_up[0]
    dos_k0_down = dos_map_spin_down[0]
    return dos_k0_up, dos_k0_down

#%%

energies, tdos_up, tdos_down = cal_tdos(data_k_xyz_dos, e_fermi)
#cal_tmr_pp_weight(data, E_fermi)

