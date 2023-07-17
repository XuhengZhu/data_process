import pprint
import os
import scipy
import numpy as np

import utils
import re

def get_test_condition(file_name):
    match = r'/(?<=\()(.+?)(?=\))/g'  # 提取测试电流数值
    angle = float(re.search(match, file_name).group())
    test_condition = {'angle': angle}
    return test_condition

# main
file_position = r'D:\Test_data\raman1'
file_list, out_path = utils.get_file_list(file_position)

pattern = r'\((\d+(\.\d+)?)\)'  # 匹配小括号中的数字，可以是整数或带有小数部分的浮点数

# 从文件名中提取角度，并创建一个带有文件名和角度的元组列表
file_angles = []
for filename in file_list:
    # 提取角度部分
    match = re.search(pattern, filename)
    if match:
        angle_str = match.group(1)
        angle = float(angle_str)
        file_angles.append((filename, angle))

# 按照角度对文件名进行排序
sorted_files = sorted(file_angles, key=lambda x: x[1])

# 将排序后的文件名存入数组
sorted_array = [file[0] for file in sorted_files]

# 打印排序后的文件名数组
pprint.pprint(sorted_array)

file_list = sorted_array

outa = r'out_smooth.txt'
new_path = os.path.join(out_path,outa)
outdata = open(new_path, 'w')
first = file_list[0]
data_name1 = file_position + "\\" + file_list[0]
data = np.loadtxt(data_name1,  dtype=float, comments='#',
                             unpack=False)
background = np.loadtxt(r"D:\Test_data\633nm-1800gr-25%-320uW-30s-background.txt",dtype=float,comments='#',unpack= False)

background = background[:,1]

background = background
background = scipy.signal.savgol_filter(background,50,3)    #将背景平滑一下
#data = data[:,0]
for file_name in file_list:
   # test_condition = get_test_condition(file_name)  # 获取测试条件
    data_name = file_position + "\\" + file_name  # 为了更改起来更方便
    data_i = np.loadtxt(data_name, dtype=float, comments='#',
                             unpack=False)  # unpack是一行是一行，一列是一列，false会把列变行

    data_i = data_i[:, 1]  # #取出第2列
    data_i = data_i
    data_i = data_i-background
    data_i = scipy.signal.savgol_filter(data_i,18,4)
    data = np.insert(data,-1,data_i,axis=1)
data[:,1:37] = data[:,1:37]-np.min(data[:,1:37])
data[:,1:37] = data[:,1:37]/np.max(data[:,1:37])    # 归一化
data = data[65:680,:]   # 去掉前后没有峰的地方
os.chdir(file_position)
np.savetxt(outdata,data, fmt ='%.4f', delimiter='  ' )








