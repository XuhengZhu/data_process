"""
MOKE 数据
找矫顽场，求loop shift
"""
import os
import re
import pprint
import utils
import numpy as np
import matplotlib.pyplot as plt

file_position = r'D:\Users\Desktop\MOKE\Only PtGd_2_I_AHE_Hx series_corr'
file_position1 ='D:\\Users\\Desktop\\MOKE\\test'
file_type = '.txt'

fit_result = []  # 单个数据文件的拟合结果，包括测试条件和对应谐波的一阶导数或者二阶倒数

file_list, out_path = utils.get_file_list(file_position)

#pprint.pprint(file_list)


def parse_current_and_field(file_name):
    # 定义正则表达式模式
    pattern = r'Hx_([+-]?[\d.]+)_I_([+-]?[\d.]+) ?(m|u)A'

    # 匹配正则表达式
    matches = re.search(pattern, file_name)

    if matches is None:
        # 未找到匹配项，返回 None
        pattern = r'Hx_([+-]?[\d.]+)_I_([+-]?[\d]+[d]+[\d]) ?(m|u)A'
        matches = re.search(pattern, file_name)

    try:
        # 获取匹配项中的电流和磁场大小
        field = float(matches.group(1).replace('d', '.'))
        current = float(matches.group(2).replace('d', '.'))
        unit = matches.group(3)

        # 转换电流大小为 mA
        if unit == 'u':
            current /= 1000
        test_condition =  {
            'field': field,
            'current': current
        }
        # 返回结果
        return test_condition
    except:
        # 解析出错，返回 None
        return None
ii = 0
for file_name in file_list:
    ii= ii%25 + 1
    if ii == 1:
        plt.figure(figsize=(14,14))
    try:
        test_condition = parse_current_and_field(file_name)  # 获取测试条件
        data_name = file_position + "\\" + file_name  # 为了更改起来更方便
        data = np.loadtxt(data_name, skiprows=1, dtype=float, comments='#',
                              unpack=False)  # unpack是一行是一行，一列是一列，false会把列变行

        data = data[:,[2,9]]
        plt.subplot(5,5,ii)
        plt.scatter(data[:,0],data[:,1],color = 'r',marker='.')

        #plt.figure(figsize=(10,10),dpi =72)
        #plt.scatter(data[:,0],data[:,1])
        #plt.show()
        data_diff = np.diff(data[:,1])
        #where = np.where(np.max(data_diff[:,1]))

        y_changes = np.diff(data[:, 1])
        y_changes_down = -1.* y_changes
        # 找到y变化最快的两个点的索引
        fastest_points_indices = np.argsort(y_changes)[-2:] + 1
        # 获取对应的x值
        fastest_points_x = data[fastest_points_indices, 0]
        x1 = np.mean(fastest_points_x)

        fastest_points_indices_down = np.argsort(y_changes_down)[-2:] + 1
        fastest_points_x = data[fastest_points_indices_down, 0]
        x2 = np.mean(fastest_points_x)

        loop_shift = (x1+x2)/2
        plt.axvline(x = loop_shift,ls ='-',c = 'green')

        plt.sbutitle(string(test_condition['field'])+'  '+string(test_condition['current']))
        test_condition['loop_shift'] = loop_shift
        fit_result.append(test_condition)
        if ii==25:
            plt.savefig(out_path+'com.png',dpi=1200,bbox_inches='tight', transparent=None)

    except:
        print(test_condition)
        print(file_name)
    #print("loop_shift", loop_shift)


#保存结果

#获取全部键值
keys = fit_result[0].keys()

# 从字典列表中提取值，构建二维数组
array = np.array([[d[key] for key in keys] for d in fit_result])
array = array[data[:, 0].argsort()]  # 按照第一列（磁场）降序排序
#array =np.around(array,4)
np.savetxt(out_path+ 'loop_shift.txt',array)
pprint.pprint(array)
#pprint.pprint(fit_result)


