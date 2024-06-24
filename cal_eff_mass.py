import numpy as np
import os
import utils


# 定义函数来解析文件并找到费米能级以上的最低能量点所在的能带
def parse_band_file(file_path):
    band_data = {}
    current_band = None
    k_points = []
    energies = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('# Band-Index'):
                # 当遇到新的能带定义时，保存上一个能带的数据
                if current_band is not None:
                    band_data[current_band] = np.column_stack((k_points, energies))
                # 重置k点和能量数据列表
                k_points = []
                energies = []
                # 设置当前能带编号
                current_band = int(line.split()[-1])
            elif line.strip() and not line.startswith('#'):
                parts = line.split()
                k = float(parts[0])
                energy = float(parts[1])
                k_points.append(k)
                energies.append(energy)

    # 添加最后一个能带的数据
    if current_band is not None:
        band_data[current_band] = np.column_stack((k_points, energies))

    return band_data

# 解析文件
#### band_data = parse_band_file(file_path)

# 找到费米能级以上的最低能量点所在的能带
def find_minimum_conduction_band(band_data):
    min_energy = float('inf')
    min_band = None
    min_band_data = None
    for band, data in band_data.items():
        # 过滤出能量大于0.1的数据
        filtered_data = data[data[:, 1] > 0.1]
        if filtered_data.size > 0:
            local_min = filtered_data[:, 1].min()
            if local_min < min_energy:
                min_energy = local_min
                min_band = band
                min_band_data = filtered_data

    return min_band, min_band_data
def find_max_valance_band(band_data):
    max_energy = float('-inf')
    max_band = None
    max_band_data = None
    for band, data in band_data.items():
        # 过滤出能量大于0.1的数据
        filtered_data = data[data[:, 1] < 0.1]
        if filtered_data.size > 0:
            local_max = filtered_data[:, 1].max()
            if local_max > max_energy:
                max_energy = local_max
                max_band = band
                max_band_data = filtered_data

    return max_band, max_band_data
#### min_band, min_band_data = find_minimum_conduction_band(band_data)
#min_band, min_band_data[:5]  # 显示最低导带的前5个数据点

# 找到最低能量点并提取其周围的9个点（如果可用）
def find_points_around_minimum(data, is_con_val, num_points=7 ):#conduction_band :0,valance_band = 1
    if is_con_val == 0:
        min_index = np.argmin(data[:, 1])
    else:
        min_index = np.argmax(data[:,1])
    half_window = num_points // 2
    start_index = max(min_index - half_window, 0)
    end_index = min(min_index + half_window + 1, data.shape[0])
    return data[start_index:end_index, :]

from scipy.constants import hbar, electron_mass, elementary_charge

from scipy.optimize import curve_fit

# 定义二次函数模型
def quadratic_model(k, a, b, c):
    return a * k**2 + b * k + c

file_position = r'D:\Users\Desktop\计算\GaSe\band_line_energy'
file_list, out_path = utils.get_file_list(file_position)
print(file_list)
eff_mass_ratio_con_min_all, eff_mass_ratio_val_max_all = [],[]
for file_name in file_list:
    #解析文件
    file_n= file_position + "\\" + file_name
    file_n
    band_data = parse_band_file(file_n)
    #找到导带
    min_band, min_band_data = find_minimum_conduction_band(band_data)
    max_band, max_band_data = find_max_valance_band(band_data)
    #拿9个点
    points_around_minimum = find_points_around_minimum(min_band_data,0)
    points_around_maximum = find_points_around_minimum(max_band_data,1)
    # 使用原始单位的K点(1/A)和能量(eV)进行拟合
    k_points_con_min = points_around_minimum[:, 0]  # K in 1/A
    k_points_val_max = points_around_maximum[:, 0]
    energies_eV_original_con_min = points_around_minimum[:, 1]  # E in eV
    energies_eV_original_val_max = points_around_maximum[:, 1]

    # 拟合并获取参数
    params_con, covariance_con = curve_fit(quadratic_model, k_points_con_min, energies_eV_original_con_min)
    params_val, covariance_val = curve_fit(quadratic_model, k_points_val_max, energies_eV_original_val_max)
    perr_con = np.sqrt(np.diag(covariance_con))  # 参数的标准误差
    # 定义单位转换常数：1 eV = 1.60218e-19 J, 1 A = 1e-10 m
    #ev_to_joules = 1.60218e-19
    angstrom_to_meters = 1e-10

    # 拟合得到的a系数转换为 J·m² 单位
    a_con_min_joules_m2 = params_con[0] * elementary_charge * angstrom_to_meters ** 2
    a_val_max_joules_m2 = -1*params_val[0] * elementary_charge * angstrom_to_meters ** 2


    # 计算有效质量，并转换为自由电子质量的比值
    eff_mass_quadr_con_min = (hbar ** 2) / (2 * a_con_min_joules_m2)
    eff_mass_quadr_val_max = (hbar ** 2) / (2 * a_val_max_joules_m2)
    eff_mass_ratio_con_min = eff_mass_quadr_con_min / electron_mass
    eff_mass_ratio_val_max = eff_mass_quadr_val_max/ electron_mass
    path_number = int(os.path.splitext(os.path.basename(file_name))[0].split('_')[1])
    eff_mass_ratio_con_min_all.append((path_number, eff_mass_ratio_con_min))
    eff_mass_ratio_val_max_all.append((path_number, eff_mass_ratio_val_max))
eff_mass_ratio_con_min_all.sort() #按路径编号排序
eff_mass_ratio_val_max_all.sort()
with open(out_path+'mass_eff_ratio_con_min.txt','w') as f:
    for path_number, eff_mass_ratio_con_min in eff_mass_ratio_con_min_all:
        f.write(f"{path_number}\t{eff_mass_ratio_con_min}\n")
out_path
with open(out_path+'mass_eff_ratio_val_max.txt','w') as f:
    for path_number, eff_mass_ratio_val_max in eff_mass_ratio_val_max_all:
        f.write(f"{path_number}\t{eff_mass_ratio_val_max}\n")
out_path


