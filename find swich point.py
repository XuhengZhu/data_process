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

file_position = r'D:\Users\Desktop\MOKE\新建文件夹\Mo_0d5_PtGd_1d5_I_AHE_Hx series_corr'
#file_position1 ='D:\\Users\\Desktop\\MOKE\\test'
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
        pattern = r'Hx_([+-]?[\d.]+)_I_([+-]?[\d]+[d]+[\d]) ?(m|u)A' #小数点用d表示
        matches = re.search(pattern, file_name)
        #print(file_name)

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
        print(file_name)
        return None

#对文件名排序
def get_sort_key(file_name):
    mag,cur = parse_current_and_field(file_name)['field'],parse_current_and_field(file_name)['current']
    print()
    return (mag,cur)

file_list = sorted(file_list, key = get_sort_key)

ii = 0
aa = 0
for file_name in file_list:
    ii= ii%25 + 1
    if ii == 1:
        plt.figure(figsize=(20,20))
    try:
        test_condition = parse_current_and_field(file_name)  # 获取测试条件
        data_name = file_position + "\\" + file_name  # 为了更改起来更方便
        data = np.loadtxt(data_name, skiprows=1, dtype=float, comments='#',
                              unpack=False)  # unpack是一行是一行，一列是一列，false会把列变行

        data = data[:,[2,9]]
        ## 删除两边由于仪器问题出现的异常值
        threshhold = 160
        indices1, indices2 = np.where(data[:, 0] > threshhold)[0], np.where(data[:, 0] < (-threshhold))[0]
        indices = np.concatenate((indices1, indices2), axis=0)
        data = np.delete(data, indices, axis=0)


        #data_diff = np.diff(data[:, 1])

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

        test_condition['loop_shift'] = loop_shift
        fit_result.append(test_condition)
        ####画图
        plt.subplot(5, 5, ii)
        plt.plot(data[:, 0], data[:, 1], 'b-', label='I: '+str(test_condition['current'])+'  Bx: '+str(test_condition['field']))
        plt.title(str(loop_shift))
        plt.legend()
        plt.axvline(x1, c='g', ls='--', lw=2)
        plt.axvline(x2, c='g', ls='--', lw=2)
        plt.axvline(loop_shift, c='r', ls='--', lw=2)
        if ii==25:
            aa +=1
            plt.savefig(out_path+str(aa)+'com.png')
            plt.close('all')

    except:
        print(test_condition)
        print(file_name)
    #print("loop_shift", loop_shift)
aa +=1
plt.savefig(out_path+str(aa)+'com.png')
plt.close('all')

#保存结果

#获取全部键值
keys = fit_result[0].keys()

# 从字典列表中提取值，构建二维数组
array = np.array([[d[key] for key in keys] for d in fit_result])
array = array[array[:, 0].argsort()]  # 按照第一列（磁场）升序排序
#array =np.around(array,4)
np.savetxt(out_path+ 'loop_shift.txt',array)
pprint.pprint(array)
#pprint.pprint(fit_result)

print(len(file_list))
print(len(fit_result))