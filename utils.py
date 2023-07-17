"""
分析数据用的函数
包括批量识别数据文件，获取测试条件
批量画图等

"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import pprint
def get_file_list(file_position):
    file_list = os.listdir(file_position)
    # 删除非.dat文件 word这些
    # print(file_list)
    for i in range(len(file_list) - 1, -1, -1):
        if not (file_list[i].endswith('.txt') or file_list[i].endswith('.dat')):  # 检查文件类型，word什么的在处理list里删了
            del file_list[i]  # 直接正序索引然后list.remove(element)的话这个迭代器里索引按照原来的数组索引递增，每一个for
            # 索引加一，但是remove后本来下一个文件的实际索引会少一，所以只会处理一半的元素、
            #  解决办法：倒序索引

    # print(file_list)
    # 输出路径
    out_path = file_position + '\\out\\'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    return file_list, out_path


#新的避免转义读取目录的方法
def get_file_list_11(file_position):
    file_list = os.listdir(file_position)
    pprint(file_list)
    # 删除非.dat文件 word这些
    # print(file_list)
    for i in range(len(file_list) - 1, -1, -1):
        if not (file_list[i].endswith('.txt') or file_list[i].endswith('.dat')):  # 检查文件类型，word什么的在处理list里删了
            del file_list[i]  # 直接正序索引然后list.remove(element)的话这个迭代器里索引按照原来的数组索引递增，每一个for
            # 索引加一，但是remove后本来下一个文件的实际索引会少一，所以只会处理一半的元素、
            #  解决办法：倒序索引

    # print(file_list)
    # 输出路径
    out =r'out'
    out_path = os.path.join(file_position,out)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    return file_list, out_path

def get_test_condition(file_name):
    # 获取测试条件
    match = r'\d+\.\d+(?=mA)'  # 提取测试电流数值
    current = float(re.search(match, file_name).group())
    # print(current)
    # print(type(current))
    # print("current"+ current)
    if not (file_name.find('1w') == -1):  # find（）返回的是找到的元素的索引，找不到返回-1，-1是True
        fit_order = 2
        harm_order = '1'
    else:
        fit_order, harm_order = 1, '2'

    if not file_name.find('down') == -1:
        Mz = 'down'
    elif not file_name.find('up') == -1:
        Mz = 'up'
    else:
        Mz = '0'  # 数据保存的时候没有表明，考虑直接删了这个数据或者画图跳过

    test_condition = {'current': current, 'harm_order': harm_order, 'Mz': Mz, 'fit_value': 0}
    return test_condition
    # 获取条件结束


#  同时测一次和二次谐波电压的数据
def get_test_condition_1(file_name):
    match = r'\d+\.\d+(?=mA)'  # 提取测试电流数值
    current = float(re.search(match, file_name).group())

    if not file_name.find('Low') == -1 or not file_name.find('down') == -1:
        Mz = 'down'
    elif not file_name.find('High') == -1 or not file_name.find('up') == -1:
        Mz = 'up'
    else:
        Mz = '0'  # 数据保存的时候没有表明，考虑直接删了这个数据或者画图跳过

    test_condition = {'current': current, 'Mz': Mz, 'fit_value': 0}
    return test_condition


def load_data(file_position, file_name):
    data_name = file_position + "\\" + file_name  # 为了更改起来更方便

    data_1_harm = np.loadtxt(data_name, skiprows=1, dtype=float, comments='#',
                             unpack=False)  # unpack是一行是一行，一列是一列，false会把列变行

    data = data_1_harm[:, 1:]  # #取出第二三列
    # 去掉大场部分，先排序,这里只做了排序
    data = data[data[:, 0].argsort()]  # 按照第一列（磁场）降序排序

    return data


#   这个两种通用

def find_first_last(data, Bmax):
    "找到小场范围磁场的数据范围"
    # first = next(x for x in data[:, 0] if x < Bmax)
    # last = next(x for x in data[:, 0] if x < (0-Bmax))
    first = next(index for index, value in enumerate(data[:, 0]) if value > (-Bmax))
    last = next(index for index, value in enumerate(reversed(data[:, 0])) if value < Bmax)
    return [first, -1 - last]


#  单独测一次和二次
def fit_and_plot(file_name, data, B_max, fit_result, out_path, test_condition):
    v_w = data[:, 1]  # 谐波电压值，暂未说明几次谐波
    B = data[:, 0]

    fit_order = 0  # 拟合阶数
    har_order = '1'  # 几次谐波，后面判断再改

    # 判断是1w还是2w
    if not (file_name.find('1w') == -1):  # 1w
        fit_order = 2
        har_order = '1'
        pos = find_first_last(data, 2 * B_max)
        B = B[pos[0]:pos[1]]
        v_w = v_w[pos[0]:pos[1]]  # 只要小场部分
    else:
        fit_order, har_order = 1, '2'
        pos = find_first_last(data, B_max)
        B = B[pos[0]:pos[1]]
        v_w = v_w[pos[0]:pos[1]]  # 只要小场部分
    # 使用二次拟合谐波电压和场的曲线 ， 得到的f1是拟合的多项式系数，是一个数组
    f1 = np.polyfit(B, v_w, fit_order)  # f1是一个数组，其中第一个是最高阶项拟合系数，次之次高阶 print("数据类型",type(f1))
    # 转换成多项式
    test_condition['fit_value'] = f1[0]

    fit_result.append(test_condition)  # 写入测试条件

    p1 = np.poly1d(f1, False, variable='B')

    v_w_fit = p1(B)  # 拟合谐波电压值

    # 绘图
    plot1 = plt.plot(B, v_w, 's', label="original values")
    plot2 = plt.plot(B, v_w_fit, "r", label="polyfit values")

    plt.xlabel('B')
    plt.ylabel("V_" + har_order + "w")
    plt.legend(loc=4)  # 指定Legend的位置在右下角
    plt.title(file_name)

    figure = plt.gcf()  # 获取当前图像
    # figure.show()

    # figure.savefig("谐波.png", dpi="figure",)  # save the current figure
    # 删除filename中的后缀名.dat
    file_name = file_name[:-4]

    figure.savefig(out_path + file_name + '.png')

    figure.clear()  # 释放内存,没有的话 会把不同曲线画在一个图里，越到后面越多曲线


#  同时测一次和二次谐波电压的数据
def fit_and_plot_1(file_name, data, B_max, fit_result, out_path, test_condition):
    v_1w = data[:, 1]  # 原.dat第三列是一次谐波，第四列是二次谐波
    v_2w = data[:, 2]
    B = data[:, 0]

    fit_order = 0  # 拟合阶数
    har_order = '0'  # 几次谐波，后面判断再改

    pos = find_first_last(data, B_max)
    B = B[pos[0]:pos[1]]
    v_1w = v_1w[pos[0]:pos[1]]  # 只要小场部分
    v_2w = v_2w[pos[0]:pos[1]]

    # 使用二次拟合谐波电压和场的曲线 ， 得到的f1是拟合的多项式系数，是一个数组
    f1 = np.polyfit(B, v_1w, 2, full=False)  # f1是一个数组，其中第一个是最高阶项拟合系数，次之次高阶 print("数据类型",type(f1))
    f2 = np.polyfit(B, v_2w, 1)

    eff_field = -0.001 * f2[0] / f1[0]  # v_2w的单位是uV，1w的单位是mV，二次的相当于数值增大了1000倍因此除以1000
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
    txtup = open(out_path + 'up.txt', 'w')
    txtdown = open(out_path + 'down.txt', 'w')
    for i in range(len(fit_result)):
        current.append(fit_result[i]['current'])
        if fit_result[i]['Mz'] == 'up':
            txtup.write(str(fit_result[i]['current']))
            txtup.write('     ')
            txtup.write(str(fit_result[i]['fit_value']))
            txtup.write('\n')
        if fit_result[i]['Mz'] == 'down':
            txtdown.write(str(fit_result[i]['current']))
            txtdown.write('     ')
            txtdown.write(str(fit_result[i]['fit_value']))
            txtdown.write('\n')

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


def caculate_DL(fit_result):
    delta_Bx = []  # 计算结果
    for i in range(len(fit_result) - 1):
        if fit_result[i]['current'] == fit_result[i + 1]['current'] and fit_result[i]['Mz'] == fit_result[i + 1]['Mz']:
            if fit_result[i]['harm_order'] == '1' and fit_result[i + 1]['harm_order'] == '2':
                delta_Bx_1 = 0 - fit_result[i + 1]['fit_value'] / fit_result[i]['fit_value']
                delta_Bx.append({'current': fit_result[i]['current'],
                                 'Mz': fit_result[i]['Mz'],
                                 'fit_value': delta_Bx_1})  # 这里为了方便把算得的场命名为fit_value，与同时测的方法一致，后面考虑改掉变量名称
    return delta_Bx


def calculat_ksi(B_Jc, tf, Ms):
    h_bar = 1.0545727e-34
    e = 1.6e-19
    t = tf * 1e-9
    Ms = Ms * 1000
    B_jc = B_Jc * 1e3 * 1e-4
    ksi = (2 * e / h_bar) * t * Ms * B_jc * (6 * 7.8e-15)
    return ksi


##  处理面内转场谐波测试的程序
###########################
###########################

def get_test_condition_rotate(file_name):
    # 获取测试条件
    match_I = r'\d+\.\d+(?=mA)'  # 提取测试电流数值
    match_B = r'\d+(?=Oe)'  # 提取测试磁场大小
    current = float(re.search(match_I, file_name).group())
    Field = float(re.search(match_B, file_name).group())

    test_condition = {'current': current, 'field': Field, 'fit_value': 0}
    return test_condition
    # 获取条件结束


def load_data_rotate(file_position, file_name):
    data_name = file_position + "\\" + file_name  # 为了更改起来更方便

    data = np.loadtxt(data_name, skiprows=0, dtype=float, comments='#',
                      unpack=False)  # unpack是一行是一行，一列是一列，false会把列变行

    data = data[data[:, 0].argsort()]  # 按照第一列（磁场）降序排序
    return data


def fit_and_plot_rotate(file_name, data, fit_result, out_path, test_condition):
    v_1w = data[:, 1]
    v_2w = data[:, 2]
    phi = data[:, 0]
    phi = np.radians(phi)

    V_PHE_popt = 0

    def R_1w(phi, V_PHE, bias):
        return V_PHE * np.sin(2 * phi) + bias

    def R_2w(phi, damp_heat, fl_oe, bias):
        return damp_heat * np.cos(phi) + 2000 * V_PHE * (2 * (np.cos(phi)) ** 3 - np.cos(phi)) * fl_oe + bias
    ###  注意单位，这里一次是mv，二次是uv

    V_PHE_popt, V_PHE_pcov = curve_fit(R_1w, phi, v_1w)
    V_PHE, bias_1w = V_PHE_popt[0], V_PHE_popt[1]
    v_2w_fit, v_2w_pcov = curve_fit(R_2w, phi, v_2w)
    damp_heat_fit, fl_oe_fit, bias_2w = v_2w_fit[0], v_2w_fit[1], v_2w_fit[2]

    # popt: 数组，参数的最佳值，以使的平方残差之和最小。f(xdata, *popt) - ydata
    # pcov: 二维阵列，popt的估计协方差。对角线提供参数估计的方差。

    v_1w_fit = R_1w(phi, V_PHE, bias_1w)
    v_2w_fit = R_2w(phi, damp_heat_fit, fl_oe_fit, bias_2w)

    plt.scatter(phi, v_1w, marker='x', lw=1, label='raw data')
    plt.plot(phi, v_1w_fit, c='r', label='fit')

    plt.xlabel('B')
    plt.ylabel("V_" + "1w")
    plt.legend(loc=4)  # 指定Legend的位置在右下角
    plt.title(file_name)

    figure = plt.gcf()  # 获取当前图像

    file_name = file_name[:-4]

    figure.savefig(out_path + file_name + '1w.png')

    figure.clear()  # 释放内存,没有的话 会把不同曲线画在一个图里，越到后面越多曲线

    plt.scatter(phi, v_2w, marker='x', lw=1, label='raw data')
    plt.plot(phi, v_2w_fit, c='r', label='fit')

    plt.xlabel('B')
    plt.ylabel("V_" + "2w")
    plt.legend(loc=4)  # 指定Legend的位置在右下角
    plt.title(file_name)

    figure = plt.gcf()  # 获取当前图像

    figure.savefig(out_path + file_name + '2w.png')

    figure.clear()  # 释放内存,没有的话 会把不同曲线画在一个图里，越到后面越多曲线

    test_condition['phe_fit'] = V_PHE_popt[0]
    test_condition['damp_heat_fit'] = damp_heat_fit
    test_condition['fl_oe_fit'] = fl_oe_fit


#   删除不要的AHE数据
def del_file_nouse(file_list):
    match_I = r'\d+\.\d+(?=mA)'  # 提取测试电流数值
    match_B = r'\d+(?=Oe)'  # 提取测试磁场大小
    for i in range(len(file_list) - 1, -1, -1):
        if not bool(re.search(match_B, file_list[i])):  # 数据文件名没有标电流值或者磁场的值的数据
            del file_list[i]  # 直接正序索引然后list.remove(element)的话这个迭代器里索引按照原来的数组索引递增，每一个for
        # 索引加一，但是remove后本来下一个文件的实际索引会少一，所以只会处理一半的元素、
        #  解决办法：倒序索引
    file_list.sort(key=lambda i: float(re.search(match_B, i).group()))  # 排序


def zip_files(file_list,union_path,file_position):
    for i in range(len(file_list) - 1):
        for j in range(i+1,len(file_list)):
            if re.search(r'\d+(?=Oe)', file_list[j]).group() == re.search(r'\d+(?=Oe)', file_list[i]).group() and \
                    re.search(r'\d+\.\d+(?=mA)', file_list[j]).group() == re.search(r'\d+\.\d+(?=mA)',
                                                                                        file_list[i]).group():
                data1 = load_data_rotate(file_position, file_list[i])
                data2 = load_data_rotate(file_position, file_list[j])
                data = np.append(data1, data2, axis=0)
                # 写入txt文件
                union = union_path + file_list[i][:-4] + '.dat' #缝合的文件名
                np.savetxt(union, data)


def reshape_test_result(fit_result,V_AHE,cu_ahe,Hk):
    result_reshape = list()
    current_list = set()

    ii = -1
    for result in fit_result:

        current_a = result['current']
        if current_a not in current_list:
            current_list.add(current_a)
            result_reshape.append(
                {'current': current_a, 'V_ahe': V_AHE * current_a / cu_ahe, 'Hk': Hk, 'damp_heat_fit': [],
                 'fl_oe_fit': [], 'field': [], 'V_phe': result['phe_fit']}) # V_ahe 是测面内磁场转角度谐波时所用电流对应的反常霍尔电压（按照线性算）
            ii = ii + 1
        result_reshape[ii]['damp_heat_fit'].append(result['damp_heat_fit'])
        result_reshape[ii]['fl_oe_fit'].append(result['fl_oe_fit'] * result['field'])  # 这里得到的FL+Oe就是以Gs为单位的最后值了
        result_reshape[ii]['field'].append(result['field'])  # 磁场单位是Oe


    return result_reshape


def fit_save_csv(result_reshape,out_path):
    for result in result_reshape:
        if not len(result['field'])-1==0:   # 只有一个数据无法拟合
            invers_field = 1/((np.array(result['field'])+result['Hk']))
            # print(invers_field)
            V_AHE =result['V_ahe'] # 当前测试电流下的V_AHE


            def damp_vs_B(invers_field, B_AD, heat):
                return V_AHE * B_AD * invers_field + heat


            damp_heat_popt,pcov = curve_fit(damp_vs_B,invers_field,result['damp_heat_fit'])
            # print(damp_heat_popt,pcov)
            damp, heat = damp_heat_popt[0],damp_heat_popt[1]

            invers_field = np.append(invers_field, 0)  # 加入无穷大场计算得到热效应
            damp_heat_fit_a = damp_vs_B(invers_field,damp,heat)
            fig_title = 'B_DL = '+str(damp)+' B_ane =' + str(heat)

            plt.scatter(invers_field[:-1],result['damp_heat_fit'],marker='o',lw = 1)
            plt.plot(invers_field,damp_heat_fit_a,c = 'r')
            plt.ylabel('V_1phi')
            plt.title(fig_title)
            figure = plt.gcf()  # 获取当前图像
            figure.savefig(out_path + str(result['current']) + 'damp.png')
            figure.clear()  # 释放内存,没有的话 会把不同曲线画在一个图里，越到后面越多曲线
        pd.set_option('display.unicode.east_asian_width',True)
        df = pd.DataFrame(index=range(0,len(result['field'])))
        for k in result.keys():
            aa=0
            df = pd.concat([df, pd.DataFrame({k: result[k]},index= range(0,len(result['field'])))], axis=1)
            aa=aa+1
        df.to_csv(out_path +str(result['current']) +'fit result.csv')


def delet_unit(file_position):
    file_list,a=get_file_list(file_position)
    out = r'out'
    for file in file_list:
        file_path = os.path.join(file_position,file)

        new_path = os.path.join(file_position,out,file)
        with open(file_path, 'r') as file:
        # 读取文件中的所有数据
            data = file.read()

         # 使用正则表达式匹配英文单位，并替换为空字符串
        data = re.sub(r'[a-zA-Z]+', '', data)

    # 将处理后的数据写回原始文件或者写入一个新文件
        with open(new_path, 'w') as file:
            file.write(data)

def delet_unit1(file_position):
    file_list, a = get_file_list(file_position)
    out = r'out'
    for file in file_list:
        file_path = os.path.join(file_position, file)
        new_path = os.path.join(file_position, out, file)

        data = np.loadtxt(file_path, skiprows=1, comments='#', dtype=str, delimiter="\t",
                          unpack=False)  # unpack是一行是一行，一列是一列，false会把列变行

        def convert_units(number, unit):
            # 将各种单位转换为对应的因子
            units = {'mV': 1e-3, 'uV': 1e-6, 'V': 1, 'A': 1, 'mA': 1e3, 'uA': 1,
                     'nA': 1e-3, 'pA': 1e-6, 'fA': 1e-9, 'ohm': 1e-3, 'kohm': 1,
                     'mohm': 1e3, 's':1}
            factor = units[unit]
            # 将字符串中的数字部分提取出来并转换为浮点数
            number = float(number)
            # 计算转换后的值并返回
            return number * factor

        import re
        newdata = np.empty([data.shape[0], data.shape[1]], dtype=float)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):

                value = data[i, j]
                # 判断是否带有单位
                if re.search(r'[a-zA-Z]', value):
                    # 提取数字和单位
                    number = re.search(r'[-+]?\d+(?:\.\d+)?', value)[0]
                    unit = re.search(r'([A-Za-z]+)', value)[0]

                    # 转换单位并将结果存储回数组中
                    x = convert_units(number, unit)
                    newdata[i, j] = x
                else:
                    newdata[i, j] = float(value)
        print(newdata)

        np.savetxt(new_path, newdata, delimiter='\t')





