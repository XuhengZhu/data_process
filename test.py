import numpy as np
'''
x = np.array([1,2,3,4,5,6,7])
y = 2*x+3
noise = np.random.rand(7)
y = y+noise
f = np.polyfit(x,y,1,full=True)
print(f)
p1 = np.poly1d(f[0],False,variable= x)
y_hat = p1(x)
print(np.sum((y_hat-y)**2))
'''
'''
print(f)
print(f[0])
'''
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import utils

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pylab as mpl

def get_test_condition_rotate(file_name):
    # 获取测试条件
    match_I = r'\d+\.\d+(?=mA)'  # 提取测试电流数值
    match_B = r'\d+(?=Oe)' #提取测试磁场大小
    current = float(re.search(match_I, file_name).group())
    Field = float(re.search(match_B, file_name).group())

    # print(current)
    # print(type(current))
    # print("current"+ current)

    test_condition = {'current': current, 'field':Field, 'fit_value': 0}
    return test_condition
    # 获取条件结束

def load_data_rotate(file_position, file_name):
    data_name = file_position + "\\" + file_name  # 为了更改起来更方便

    data = np.loadtxt(data_name, skiprows=1, dtype=float, comments='#',
                             unpack=False)  # unpack是一行是一行，一列是一列，false会把列变行


    data = data[data[:, 0].argsort()]  # 按照第一列（磁场）降序排序

    return data

def fit_and_plot_rotate(file_name, data,  fit_result, out_path, test_condition):

    v_1w = data[:, 1]
    v_2w = data[:, 2]
    phi = data[:, 0]
    phi = np.radians(phi)

    R_PHE_popt = 0

    def R_1w(phi, R_PHE,bias):
        return R_PHE * np.sin(2*phi) + bias

    def R_2w(phi,damp_heat,fl_oe,bias):
        return damp_heat * np.cos(phi) + 2*R_PHE * (2* (np.cos(phi))**3 -np.cos(phi))*fl_oe +bias

    R_PHE_popt, R_PHE_pcov = curve_fit(R_1w, phi, v_1w)
    R_PHE, bias_1w = R_PHE_popt[0], R_PHE_popt[1]
    v_2w_fit , v_2w_pcov = curve_fit(R_2w,phi,v_2w)
    damp_heat_fit, fl_oe_fit, bias_2w = v_2w_fit[0], v_2w_fit[1], v_2w_fit[2]

    # popt: 数组，参数的最佳值，以使的平方残差之和最小。f(xdata, *popt) - ydata
    # pcov: 二维阵列，popt的估计协方差。对角线提供参数估计的方差。

    v_1w_fit = R_1w(phi, R_PHE,bias_1w)
    v_2w_fit = R_2w(phi,damp_heat_fit,fl_oe_fit,bias_2w)

    plt.scatter(phi,v_1w,marker = 'x',lw = 1, label = 'raw data')
    plt.plot(phi,v_1w_fit, c = 'r',label = 'fit')



    plt.xlabel('B')
    plt.ylabel("V_" + "1w")
    plt.legend(loc=4)  # 指定Legend的位置在右下角
    plt.title(file_name)

    figure = plt.gcf()  # 获取当前图像

    file_name = file_name[:-4]

    figure.savefig(out_path + file_name + '1w.png')

    figure.clear()  # 释放内存,没有的话 会把不同曲线画在一个图里，越到后面越多曲线

    plt.scatter(phi,v_2w,marker = 'x',lw = 1, label = 'raw data')
    plt.plot(phi,v_2w_fit, c = 'r',label = 'fit')

    plt.xlabel('B')
    plt.ylabel("V_" + "2w")
    plt.legend(loc=4)  # 指定Legend的位置在右下角
    plt.title(file_name)

    figure = plt.gcf()  # 获取当前图像



    figure.savefig(out_path + file_name + '2w.png')

    figure.clear()  # 释放内存,没有的话 会把不同曲线画在一个图里，越到后面越多曲线

    test_condition['phe_fit'] = R_PHE_popt
    test_condition['damp_heat_fit'] = damp_heat_fit
    test_condition['fl_oe_fit'] = fl_oe_fit

import operator
import utils
import pprint
file_position = 'D:\\Test_data\\zhushouguo\\Orbital Hall\\RuO\\4-Ru8Gd2Co6\\test'
file_type = '.dat'


fit_result = []  # 单个数据文件的拟合结果，包括测试条件和对应谐波的一阶导数或者二阶倒数

file_list, out_path = utils.get_file_list(file_position)


for file_name in file_list:
    test_condition = get_test_condition_rotate(file_name)  # 获取测试条件

    data = load_data_rotate(file_position, file_name)  # 从。dat文件载入数据并给数据排序

    #pos = #utils.find_first_last(data, B_max)  # 找到小场范围

    fit_and_plot_rotate(file_name, data,  fit_result, out_path, test_condition)

    fit_result.append(test_condition)


pprint.pprint(fit_result)