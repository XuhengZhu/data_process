import numpy as np
import matplotlib.pyplot as plt
import tmr_vdos
import pickle
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
            kx, ky = round(kx, 6), round(ky, 6)
            i += 1  # 移动到能带数据行
            band_info = []

            # 继续读取直到遇到下一个K点信息或文件结束
            while i < len(lines) and len(lines[i].split()) == 3:
                band_line = lines[i].strip().split()
                band_index, eigenval, occ_down = [int(band_line[0]), float(band_line[1]), float(band_line[2])]
                band_info.append((band_index, eigenval, occ_down))
                i += 1
            k_points.append((kx, ky, kz, weight))
            bands.append(band_info)
        else:
            # 非预期行，跳过
            i += 1

    return k_points, bands

e_fermi = 1.941
filepath = r'D:\Users\Desktop\计算\GaSe\EIGENVAL_GGA_txt'
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
np.savetxt('data/kxkykzweight.txt', k_points)

#%%
# 找cbm
def cbm_cal(k_points, bands, e_fermi):
    cbm_map ={}
    cbms = []
    cbma = 10
    for i, (kx, ky, _, _) in enumerate(k_points):
        if (kx,ky) not in cbm_map:
            cbm_map[(kx,ky)] = cbma-e_fermi
        for band in bands[i]:
            _, energe, _ = band
            if band[2]<0.50:
                if energe-e_fermi < cbm_map[(kx,ky)]:
                    cbm_map[(kx,ky)] = energe-e_fermi
    print(len(cbm_map))
    return cbm_map

cbm_map = cbm_cal(k_points,bands,e_fermi)
d_save = open('data/cbm_map.pkl','wb')
pickle.dump(cbm_map,d_save)
d_save.close()
with open('data/cbm_map.pkl','rb') as f:
    loaded_cbm_map = pickle.load(f)
print(loaded_cbm_map)
np.save('data/cbm_map.npy',cbm_map)

print((cbm_map))

#%% 旋转和对称变换
def rotate(theta, cbm_map)
    kx = cbm_map

#%%
def draw_cbm_map(cbm_map):
    kx = []
    ky = []
    cbm = []

    print(cbm_map)
    for key, cbm1 in cbm_map.items():
        kx.append(key[0])
        ky.append(key[1])
        cbm.append(cbm1)
    print(len(kx))
    print(len(cbm))
    print(len(ky))
    kx = np.array(kx)
    ky = np.array(ky)
    cbm = np.array(cbm)
    fig = plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(kx, ky, cbm)
    '''
    #x, y = np.meshgrid(kx,ky)
    ax = plt.axes(projection='3d')
    #ax.plot_surface(x, y,cbm, cmap='rainbow')
    ax.contour(kx, ky, cbm,zdim='z',offset = 2.6, cmap='rainbow')
    '''
    plt.show()

draw_cbm_map(cbm_map)



