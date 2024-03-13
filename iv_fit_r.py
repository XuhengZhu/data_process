import os
import re
import pprint
import utils
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']	# 显示中文 非衬线字体为黑体
plt.rcParams['axes.unicode_minus'] = False	#显示负号

file_position = r'D:\Users\Desktop\重要同步\FGaT_GaSe_FGeT\2.2FGaTGaSeFGeT\c\iv'

file_type = '.txt'

fit_result = []  # 单个数据文件的拟合结果，包括测试条件和对应谐波的一阶导数或者二阶倒数

file_list, out_path = utils.get_file_list(file_position)
out_path = file_position + '\\out\\'


for file_name in file_list:
    data_name = file_position + "\\" + file_name  # 为了更改起来更方便

    data = np.loadtxt(data_name, dtype=float, comments='#',
                      unpack=False)  # unpack是一行是一行，一列是一列，false会把列变行,range(1,5)为处理iv

    # 去掉大场部分，先排序,这里只做了排序
    data = data[data[:, 0].argsort()]  # 按照第一列（dianya）降序排序
    unique_x = np.unique(data[:, 0])
    #iv数据有6列，tmr数据有7列，其中第二列是多的为时间，最后一列都为电阻
    '''
    data[:,0] = -1*data[:,0]
    data[:,1] = -1*data[:,1]
    '''
    # 创建空列表存储结果
    '''
    result = []
    for x in unique_x:
        y_values = data[data[:, 0] == x, 1:]  # 选择除第一列外的所有列
        average_y = np.mean(y_values, axis=0)
        result.append([x] + list(average_y))
    '''
    f2 = np.polyfit(data[:,0],data[:,1], 1)
    data = result
    np.savetxt(out_path + file_name, data)
