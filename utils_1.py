#  同时测一次和二次谐波用

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import utils
#  同时测一次和二次谐波电压的数据
def get_test_condition_1(file_name):
    match = r'\d+\.\d+(?=mA)'  # 提取测试电流数值
    current = float(re.search(match, file_name).group())

    if not file_name.find('down') == -1:
        Mz = 'down'
    elif not file_name.find('up') == -1:
        Mz = 'up'
    else:
        Mz = '0'  # 数据保存的时候没有表明，考虑直接删了这个数据或者画图跳过

    test_condition = {'current': current, 'Mz': Mz, 'fit_value': 0}
    return test_condition


#  同时测一次和二次谐波电压的数据
def fit_and_plot_1(file_name, data, B_max, fit_result, out_path, test_condition):
    v_1w = data[:, 1]  # 原.dat第三列是二次，第四列是一次
    v_2w = data[:, 2]
    B = data[:, 0]

    fit_order = 0  # 拟合阶数
    har_order = '0'  # 几次谐波，后面判断再改

    pos = utils.find_first_last(data, B_max)
    B = B[pos[0]:pos[1]]
    v_1w = v_1w[pos[0]:pos[1]]  # 只要小场部分
    v_2w = v_2w[pos[0]:pos[1]]

    # 使用二次拟合谐波电压和场的曲线 ， 得到的f1是拟合的多项式系数，是一个数组
    f1 = np.polyfit(B, v_1w, 2)  # f1是一个数组，其中第一个是最高阶项拟合系数，次之次高阶 print("数据类型",type(f1))
    f2 = np.polyfit(B, v_2w, 1)

    eff_field = -0.002 * f2[0] / f1[0]  # v_2w的单位是uV，1w的单位是mV，二次的相当于数值增大了1000倍因此除以1000
    test_condition['fit_value'] = eff_field

    # 下面四行是绘图用的数据
    p1 = np.poly1d(f1, False, variable='B')
    p2 = np.poly1d(f2, False, variable='B')
    v_1w_fit = p1(B)  # 拟合谐波电压值
    v_2w_fit = p2(B)
    #  绘图 1w
    plot1 = plt.plot(B, v_1w, 's', label="original values")
    plot2 = plt.plot(B, v_1w_fit, "r", label="polyfit values")

    plt.xlabel('B')
    plt.ylabel("V_" + "1w")
    plt.legend(loc=4)  # 指定Legend的位置在右下角
    plt.title(file_name)

    figure = plt.gcf()  # 获取当前图像

    file_name = file_name[:-4]

    figure.savefig(out_path + file_name + '.png')

    figure.clear()  # 释放内存,没有的话 会把不同曲线画在一个图里，越到后面越多曲线

    plot1 = plt.plot(B, v_2w, 's', label="original values")
    plot2 = plt.plot(B, v_2w_fit, "r", label="polyfit values")

    plt.xlabel('B')
    plt.ylabel("V_" + "2w")
    plt.legend(loc=4)  # 指定Legend的位置在右下角
    plt.title(file_name)

    figure = plt.gcf()  # 获取当前图像

    file_name = file_name[:-4]

    figure.savefig(out_path + file_name + '.png')

    figure.clear()  # 释放内存,没有的话 会把不同曲线画在一个图里，越到后面越多曲线

#  画有效场对于电流的图,同时测一次和二次用
def plot_field(fit_result, out_path):
    current, eff_field = [0, ], [0, ]
    for i in range(len(fit_result) - 1):
        current.append(fit_result[i]['current'])
        eff_field.append(fit_result[i]['fit_value'])
    plot1 = plt.plot(current, eff_field, 's', label="eff_field_DL")

    plt.xlabel('I')
    plt.ylabel("B_DL")
    plt.legend(loc=4)  # 指定Legend的位置在右下角
    plt.title('B_DL vs I')
    figure = plt.gcf()  # 获取当前图像
    file_name = 'Mz =' + fit_result[0]['Mz'] + 'result'
    figure.savefig(out_path + file_name + '.png')
    figure.clear()  # 释放内存,没有的话 会把不同曲线画在一个图里，越到后面越多曲线
