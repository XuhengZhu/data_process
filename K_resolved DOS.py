import numpy as np
import matplotlib.pyplot as plt
import tmr_vdos
from scipy.integrate import simps
import test5
import pickle

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
k_points, bands = read_eigenval_file_corrected(filepath,e_fermi)


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
k_points, k_points_abs = convert_to_cartesian(k_points)
print(k_points)
print(k_points_abs)
np.savetxt('data/kxkykzweight.txt', k_points)


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
            k_points_pro.append((kx,ky))
            dos_map_spin_up_pro[(kx,ky)] = np.zeros(len(energies))
            dos_map_spin_down_pro[(kx,ky)] = np.zeros(len(energies))
        dos_map_spin_up_pro[(kx,ky)] += dos_map_spin_up[i]*weigh # 后面不用乘以这个权重了对于prodos
        dos_map_spin_down_pro[(kx,ky)] += dos_map_spin_down[i]*weigh

    # 将字典转换为列表方便处理
    #dos_map_spin_up_down
    return (energies, dos_map_spin_up, dos_map_spin_down), (energies, dos_map_spin_up_pro, dos_map_spin_down_pro),k_points_pro



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

data, data_pro, k_points_pro = calculate_k_resolved_dos(k_points, bands, energy_range, delta_e, sigma)
data = (k_points,) + data  # 向元组中添加元素


#%% 对称变换函数，补全k点
def cal_k_sym(k_points_pro):
    #k_points_all = k_points_pro#
    # 这个只是复制了引用给他，后续在k_points_all.append时会动态改变K_points_pro的，越来越长，最后爆掉内存
    k_points_all = k_points_pro[:] #这样才是赋值，创建新列表
    kx, ky = [], []
    #旋转变换
    for i in range(1,6):
        theta = np.pi * i / 3
        print(i)
        a=1
        for j, (kx0,ky0) in enumerate(k_points_pro):
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
    np.savetxt('data/tunnelp_kxky_u.txt', k_points_all2)
    plt.scatter(kx,ky)
    plt.show()
    plt.clf()
    return k_points_all2

k_points_all2 = cal_k_sym(k_points_pro)

#%% 读取导带最低点信息文件
with open('data/474713cbm_map.pkl','rb') as f:
    cbm_map = pickle.load(f)
with open('data/474713vbm_map.pkl','rb') as d:
    vbm_map = pickle.load(d)
print(cbm_map)
#print(cbm_map[(0.0,0.0)])

# %%
# data = (k_points,energies, dos_map_spin_up, dos_map_spin_down)
def weight_kpoint(k_point, U, Ef,m_eff, E=0, a=3.98E-10, d=5E-9):  # 给归一化的相对坐标
    eV = 1.6022E-19
    h_bar = 1.05257E-34
    m = 9.1E-31
    m_eff1 = m_eff
    U1 = U
    kx, ky = k_point[0] / a, k_point[1] / a
    kapa = (np.sqrt(2 * m * m_eff1 * eV * (U1 - Ef + E) + (h_bar ** 2) * (kx ** 2 + ky ** 2))) / h_bar
    kappa_00 = (np.sqrt(2 * m * m_eff1 * eV * (1.00 - Ef + E))) / h_bar #gamma点
    weight_00 = np.exp(-2 * kappa_00 * d)
    weight = np.exp(-2 * kapa * d) / weight_00
    return weight


### 绘出二维平面内K点对应权重大小
def draw_weights(k_points_pro, cbm_map):
    weight = []
    cbm_mapa = cbm_map
    kx, ky, weight_sym = [], [], []
    weight_k_only = [] # 只考虑K平行带来的隧穿，即不同k点的势垒高度认为相同
    # weight_sym 计算完投影的时候不需要，在计算投影的时候用就行
    for k_point in k_points_pro:
        U = cbm_mapa[(k_point[0], k_point[1])]
        weight.append(weight_kpoint(k_point, U-1.0326, 0, m_eff, 0, a=3.98E-10, d=5E-9))
        weight_k_only.append(weight_kpoint(k_point,1.00,0, m_eff, 0, a=3.98E-10, d=5E-9))
        kx.append(k_point[0])
        ky.append(k_point[1])
        # weight_sym.append(k_point[3])
    print(weight)
    tunnelp_kxky = np.vstack([kx, ky, weight]).T
    tunnelp_k = np.vstack([kx,ky,weight_k_only]).T
    np.savetxt('data/tunnelp_kxky_u.txt', tunnelp_kxky)
    np.savetxt('data/tunnelp_kxky_only.txt', tunnelp_k)
    print(np.vstack([kx, ky, weight]).T)
    kx_all = k_points_all2[:,0]
    ky_all = k_points_all2[:,1]
    weight_ku = weight[:]
    weight_k = weight_k_only[:]
    weight_all = weight[:]
    weight_k_only_all = weight_k_only[:]
    for i in range(1,12):
        # weight_all.append(weight), 这样是将列表作为一个元素添加到后面，错了
        weight_all.extend(weight)
        weight_k_only_all.extend(weight_k_only)
        print(i)

    #kx_fine, ky_fine, weight_matr = test5.smooth_data(kx_all, ky_all, weight_all,0.75)
    #kx1,ky1,weight_k_only_matr = test5.smooth_data(kx_all,ky_all,weight_k_only_all,0.75)

    #test5.draw(kx_fine, ky_fine, weight_matr,0.75,'weight_k+u')

    #test5.draw(kx_fine, ky_fine, weight_k_only_matr,0.75,'weight_k-only')
    return (kx, ky, weight_ku,weight_k)


kx, ky, weight_ku, weight_k = draw_weights(k_points_pro, cbm_map)
kxpp, kypp, weight_ku_pp, weight_k_pp = draw_weights(k_points, cbm_map)

#%%
def draw_p_dos(data_pro, kx,ky,e):

    energies = data_pro[0]
    e_close = np.isclose(energies, e, atol=1e-8)  # atol是容忍度
    index_e = np.where(e_close)[0][0]
    print(index_e)
    dos_map_spin_up = data_pro[1]
    dos_map_spin_down = data_pro[2]
    dos_down_e = np.zeros(len(kx))
    dos_up_e = np.zeros(len(kx))
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
    kx_all = k_points_all2[:, 0]
    ky_all = k_points_all2[:, 1]
    kx1,ky1,dos_up_matr = test5.smooth_data(kx_all,ky_all,dos_up_e_all,0.75)
    kx1, ky1, dos_down_matr = test5.smooth_data(kx_all, ky_all, dos_down_e_all, 0.75)

    test5.draw(kx1, ky1, dos_up_matr,0.75,'dos_up e='+str(e))
    test5.draw(kx1,ky1,dos_down_matr,0.75,'dos_down e='+str(e))
    return()
for i in (-0.5,-0.25,0,0.25,0.5):
    draw_p_dos(data_pro,kx,ky,i)

# %%
def weight_kpoints(k_points, Us, Ef,m_eff, Es=0):
    h_bar = 1.05257E-34
    m = 9.1E-31
    m_eff = m_eff
    weights = np.zeros(len(k_points))
    for i in len(k_points):
        kx, ky = k_points[i][0], k_points[i][1]
        weights[i] = np.exp(np.sqrt(Us[i] - Ef + Es + (h_bar ** 2) * (kx ** 2 + ky ** 2) / (2 * m_eff * m)))
    return weights


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
cal_tdos(data,0)

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
cal_dos_rough(data,0)

#%% 点对点处理计算 tmr方案
def cal_tmr_pp_weight(data, E_fermi):
    k_points, energies, dos_map_spin_up, dos_map_spin_down = data[0], data[
        1], data[2], data[3]
    vb_range = np.linspace(-1.5, 1.5, 500)
    conductance_p_total, conductance_ap_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
    conductance_p_k_total, conductance_ap_k_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
    # 先计算电导，再将电导加权这样才正确
    for i, (_, _, _, _) in enumerate(k_points):
        conductance_p, conductance_ap = tmr_vdos.cal_conduct(vb_range,energies, dos_map_spin_up[i],
                                                                              dos_map_spin_down[i])
        conductance_p_ku = k_points[i][3] * weight_ku_pp[i] *conductance_p
        conductance_ap_ku = k_points[i][3] * weight_ku_pp[i] *conductance_ap
        conductance_p_total += conductance_p_ku
        conductance_ap_total += conductance_ap_ku

        conductance_p_k = k_points[i][3] * weight_k_pp[i] * conductance_p
        conductance_ap_k = k_points[i][3] * weight_k_pp[i] * conductance_ap
        conductance_p_k_total += conductance_p_k
        conductance_ap_k_total += conductance_ap_k
    tmr = np.where(conductance_ap_total != 0, (conductance_p_total - conductance_ap_total) / (conductance_ap_total+epsilon), 0)
    tmr_k = np.where(conductance_ap_k_total != 0,(conductance_p_k_total - conductance_ap_k_total)/(conductance_ap_k_total+epsilon), 0)


    '''
    plt.plot(vb_range,conductance_p_total, label='p')   
    plt.plot(vb_range,conductance_ap_total, label='ap')
    plt.legend()
    plt.show()
    # 绘制图表
    plt.figure(figsize=(8, 6))
    '''
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(vb_range, tmr, label='TMR vs Vb')
    plt.xlabel('Bias Voltage (V)')
    plt.ylabel('TMR')
    plt.title('TMR pp k+u')
    plt.grid(True)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(vb_range, conductance_p_k_total, label='p')
    plt.plot(vb_range, conductance_ap_k_total, label='ap')
    plt.legend()
    plt.show()
    # 绘制图表
    plt.figure(figsize=(8, 6))
    plt.plot(vb_range, tmr_k, label='TMR vs Vb')
    plt.xlabel('Bias Voltage (V)')
    plt.ylabel('TMR')
    plt.title('TMR pp k-only')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.clf()
cal_tmr_pp_weight(data,e_fermi)
mass_eff_ratio_path = r"D:\Users\Desktop\计算\GaSe\band_line_energy\out\mass_eff_ratio.dat"
mass_eff_ratio = np.loadtxt(mass_eff_ratio_path,usecols=1, unpack=True)

# %% 投影到kxky平面内处理
# def cal_tmr_pro_weight(data_pro, E_fermi):
#     vb_range = np.linspace(-1.5, 1.5, 1500)
#     conductance_p_pro_total, conductance_ap_pro_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
#     conductance_p_pro_k_total, conductance_ap_pro_k_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
#     energies, dos_up_pro, dos_down_pro =  data_pro[0], data_pro[1], data_pro[2]
#     current_p_distr= []
#     current_ap_distr= []
#     kxky = []
#     mass_n = 0
#     for (kx,ky) in dos_down_pro:
#         mass_n = mass_n + 1
#         #print((kx,ky))
#         conductance_p_pro, conductance_ap_pro = tmr_vdos.cal_conduct(vb_range, energies, dos_up_pro[(kx,ky)], dos_down_pro[(kx,ky)])
#         # 电导分布
#         cudp,cudap = [],[]
#         a=0
#         for i in (752,800,850,900,950,1000,1050,1100,1150,1200,1250):
#             cudp.append(conductance_p_pro[i])
#             cudap.append(conductance_ap_pro[i])
#         #print(cudp)
#         current_p_distr.append(cudp)
#         #print(current_p_distr)
#         current_ap_distr.append(cudap)
#         kxky.append([kx,ky])
#         U = cbm_map[(kx,ky)]
#         #print(U)
#         #print(weight_kpoint((kx,ky,0,0),U-1.03 , 0,m_eff, 0, a=3.98E-10, d=5E-9))
#         conductance_p_pro_ku = conductance_p_pro * weight_kpoint((kx,ky,0,0),U-1.03 , 0,0.1, 0, a=3.98E-10, d=5E-9)
#         conductance_ap_pro_ku = conductance_ap_pro * weight_kpoint((kx, ky, 0, 0), U-1.03, 0,0.1, 0, a=3.98E-10, d=5E-9)
#         conductance_p_pro_total += conductance_p_pro_ku
#         conductance_ap_pro_total += conductance_ap_pro_ku
#
#         conductance_p_pro_k = conductance_p_pro * weight_kpoint((kx,ky,0),1,0,m_eff,0,a=3.98E-10, d=5E-9)
#         conductance_ap_pro_k = conductance_ap_pro * weight_kpoint((kx,ky,0),1,0,m_eff,0,a=3.98E-10, d=5E-9)
#         conductance_p_pro_k_total +=conductance_p_pro_k
#         conductance_ap_pro_k_total += conductance_ap_pro_k
#     tmr = np.where(conductance_ap_pro_total !=0, (conductance_p_pro_total-conductance_ap_pro_total)/(conductance_ap_pro_total+epsilon), 0)
#     tmr_k = np.where(conductance_ap_pro_k_total !=0,(conductance_p_pro_k_total-conductance_ap_pro_k_total)/(conductance_ap_pro_k_total+epsilon),0)
#     '''
#     plt.figure(figsize = (8, 6))
#     plt.plot(vb_range, conductance_p_pro_total, label='p')
#     plt.plot(vb_range, conductance_ap_pro_total, label='ap')
#     plt.legend()
#     plt.show()
#     plt.close()
#     '''
#     #绘制电流分布
#     cudp1 = np.array(current_p_distr)
#     cudap1 = np.array(current_ap_distr)
#     cudp2 = np.tile(cudp1,(12,1))
#     cudap2 = np.tile(cudap1,(12,1))
#
#     kxky = np.array(kxky)
#     kxall = k_points_all2[:,0]
#     kyall = k_points_all2[:,1]
#     fig,axs = plt.subplots(2,11,figsize=(40,200))
#     for i in range(0,11):
#         cup,cuap = cudp2[:,i],cudap2[:,i]
#         kx1, ky1, current_p_ma = test5.smooth_data(kxall, kyall, cup, 0.75)
#         kx1, ky1, current_ap_ma = test5.smooth_data(kxall, kyall,cuap, 0.75)
#         k_range=0.75
#         ax1 = axs[0,i]
#         ax2 = axs[1,i]
#         im1 = ax1.imshow(current_p_ma, extent=(-1 * k_range, k_range, -1 * k_range, k_range), cmap='viridis',
#                        interpolation='bilinear', origin='lower')
#         im2 = ax2.imshow(current_ap_ma, extent=(-1 * k_range, k_range, -1 * k_range, k_range), cmap='viridis',
#                        interpolation='bilinear', origin='lower')
#         ax1.set_title('p')
#         ax2.set_title('ap')
#     plt.show(block=True)
#     plt.clf()
#     # 绘制tmr
#     plt.figure(figsize = (12,6))
#     plt.subplot(121)
#     plt.plot(vb_range, tmr, label='TMR vs Vb')
#     plt.xlabel('Bias Voltage (V)')
#     plt.ylabel('TMR')
#     plt.title('TMR pro k+u')
#     plt.grid(True)
#     plt.subplot(122)
#     plt.plot(vb_range,tmr_k,label='TMR vs')
#     plt.xlabel('Bias Voltage (V)')
#     plt.ylabel('TMR')
#     plt.title('TMR pro k-only')
#     plt.legend()
#     plt.show(block=True)
def cal_tmr_pro_weight(data_pro, E_fermi):
    vb_range = np.linspace(-1.5, 1.5, 1500)
    conductance_p_pro_total, conductance_ap_pro_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
    conductance_p_pro_k_total, conductance_ap_pro_k_total = np.zeros(len(vb_range)), np.zeros(len(vb_range))
    energies, dos_up_pro, dos_down_pro =  data_pro[0], data_pro[1], data_pro[2]
    current_p_distr= []
    current_ap_distr= []
    kxky = []
    mass_n = 0
    for (kx,ky) in dos_down_pro:

        mass_n = mass_n + 1
        if kx ** 2 + ky ** 2 <= 0.13:
            #print((kx,ky))
            conductance_p_pro, conductance_ap_pro = tmr_vdos.cal_conduct(vb_range, energies, dos_up_pro[(kx,ky)], dos_down_pro[(kx,ky)])
            # 电导分布
            cudp,cudap = [],[]
            a=0
            for i in (752,800,850,900,950,1000,1050,1100,1150,1200,1250):
                cudp.append(conductance_p_pro[i])
                cudap.append(conductance_ap_pro[i])
            #print(cudp)
            current_p_distr.append(cudp)
            #print(current_p_distr)
            current_ap_distr.append(cudap)
            kxky.append([kx,ky])
            U = cbm_map[(kx,ky)]
            #print(U)
            #print(weight_kpoint((kx,ky,0,0),U-1.03 , 0,m_eff, 0, a=3.98E-10, d=5E-9))
            conductance_p_pro_ku = conductance_p_pro
            conductance_ap_pro_ku = conductance_ap_pro
            conductance_p_pro_total += conductance_p_pro_ku
            conductance_ap_pro_total += conductance_ap_pro_ku

            conductance_p_pro_k = conductance_p_pro
            conductance_ap_pro_k = conductance_ap_pro
            conductance_p_pro_k_total +=conductance_p_pro_k
            conductance_ap_pro_k_total += conductance_ap_pro_k
    tmr = np.where(conductance_ap_pro_total !=0, (conductance_p_pro_total-conductance_ap_pro_total)/(conductance_ap_pro_total+epsilon), 0)
    tmr_k = np.where(conductance_ap_pro_k_total !=0,(conductance_p_pro_k_total-conductance_ap_pro_k_total)/(conductance_ap_pro_k_total+epsilon),0)
    '''
    plt.figure(figsize = (8, 6))
    plt.plot(vb_range, conductance_p_pro_total, label='p')
    plt.plot(vb_range, conductance_ap_pro_total, label='ap')
    plt.legend()
    plt.show()
    plt.close()
    '''
    #绘制电流分布
    cudp1 = np.array(current_p_distr)
    cudap1 = np.array(current_ap_distr)
    cudp2 = np.tile(cudp1,(12,1))
    cudap2 = np.tile(cudap1,(12,1))

    kxky = np.array(kxky)
    # kxall = k_points_all2[:,0]
    # kyall = k_points_all2[:,1]
    # fig,axs = plt.subplots(2,11,figsize=(40,200))
    # for i in range(0,11):
    #     cup,cuap = cudp2[:,i],cudap2[:,i]
    #     kx1, ky1, current_p_ma = test5.smooth_data(kxall, kyall, cup, 0.75)
    #     kx1, ky1, current_ap_ma = test5.smooth_data(kxall, kyall,cuap, 0.75)
    #     k_range=0.75
    #     ax1 = axs[0,i]
    #     ax2 = axs[1,i]
    #     im1 = ax1.imshow(current_p_ma, extent=(-1 * k_range, k_range, -1 * k_range, k_range), cmap='viridis',
    #                    interpolation='bilinear', origin='lower')
    #     im2 = ax2.imshow(current_ap_ma, extent=(-1 * k_range, k_range, -1 * k_range, k_range), cmap='viridis',
    #                    interpolation='bilinear', origin='lower')
    #     ax1.set_title('p')
    #     ax2.set_title('ap')
    plt.show(block=True)
    plt.clf()
    # 绘制tmr
    plt.figure(figsize = (12,6))
    plt.subplot(121)
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
    plt.show(block=True)

cal_tmr_pro_weight(data_pro, 0)


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
        a=0
        for i in (752,800,900,1000,1100,1200):
            cudp.append(conductance_p_pro[i])
            cudap.append(conductance_ap_pro[i])
        print(cudp)
        current_p_distr.append(cudp)
        #print(current_p_distr)
        current_ap_distr.append(cudap)
        kxky.append([kx,ky])

        #print(weight_kpoint((kx,ky,0,0),U-1.03 , 0,m_eff, 0, a=3.98E-10, d=5E-9))
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
    #绘制电流分布
    cudp1 = np.array(current_p_distr)
    cudap1 = np.array(current_ap_distr)
    cudp2 = np.tile(cudp1,(12,1))
    cudap2 = np.tile(cudap1,(12,1))

    kxky = np.array(kxky)
    kxall = k_points_all2[:,0]
    kyall = k_points_all2[:,1]
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
    '''
    plt.subplot(122)
    plt.plot(vb_range,tmr_k,label='TMR vs')
    plt.xlabel('Bias Voltage (V)')
    plt.ylabel('TMR')
    plt.title('TMR pro k-only')
    plt.legend()
    '''
    plt.show(block=True)

cal_tmr_pro_weight_acc(data_pro, 0)

# 计算单gamma点dos及相关tmr
def cal_dos_k0(k_points, energies, dos_map_spin_up, dos_map_spin_down, E_fermi):
    dos_k0_up = dos_map_spin_up[0]
    dos_k0_down = dos_map_spin_down[0]
    return dos_k0_up, dos_k0_down

#%%

energies, tdos_up, tdos_down = cal_tdos(data, E_fermi)
#cal_tmr_pp_weight(data, E_fermi)

'''
#dos_k0_up, dos_k0_down = cal_dos_k0(data)
# %%
plt.figure(figsize=(5, 5))
plt.plot(energies, tdos_up, linewidth=1, label='tdos_up')
plt.plot(energies, 0 - tdos_down, linewidth=1, label='tdos_down')
plt.xlim(-3, 3), plt.ylim(-30, 30)
plt.legend()
plt.show()
# %%
# gamma 点
plt.figure(figsize=(5, 5))
plt.plot(energies, dos_k0_up, linewidth=1, label='tdos_up')
plt.plot(energies, 0 - dos_k0_down, linewidth=1, label='tdos_down')
plt.xlim(-3, 3), plt.ylim(-30, 30)
plt.legend()
plt.show()
# %%
data = np.vstack([energies, tdos_up, tdos_down]).T
# data_tmr_w =np.vstack([energies, tdos_tmr_up, tdos_tmr_down]).T
# 指定保存路径，更改为你的目标路径


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