import os
import re
import numpy as np
import matplotlib.pyplot as plt


def get_file_list(file_position):
    file_list = os.listdir(file_position)
    # 删除非.dat文件 word这些
    # print(file_list)
    for i in range(len(file_list)-1,-1,-1):
        if not file_list[i].endswith('.dat'):  # 检查文件类型，word什么的在处理list里删了
            del file_list[i]  # 直接正序索引然后list.remove(element)的话这个迭代器里索引按照原来的数组索引递增，每一个for
            # 索引加一，但是remove后本来下一个文件的实际索引会少一，所以只会处理一半的元素、
                              #  解决办法：倒序索引

    # print(file_list)
    # 输出路径
    out_path = file_position + '\\out\\'
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

    if not file_name.find('down')== -1:
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

    if not file_name.find('down') == -1:
        Mz = 'down'
    elif not file_name.find('up') == -1:
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
    # 去掉大场部分，先排序
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
        pos = find_first_last(data, 2*B_max)
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
    v_1w = data[:, 1]  # 原.dat第三列是二次，第四列是一次
    v_2w = data[:, 2]
    B = data[:, 0]

    fit_order = 0  # 拟合阶数
    har_order = '0'  # 几次谐波，后面判断再改

    pos = find_first_last(data, B_max)
    B = B[pos[0]:pos[1]]
    v_1w = v_1w[pos[0]:pos[1]]  # 只要小场部分
    v_2w = v_2w[pos[0]:pos[1]]

    # 使用二次拟合谐波电压和场的曲线 ， 得到的f1是拟合的多项式系数，是一个数组
    f1 = np.polyfit(B, v_1w, 2)  # f1是一个数组，其中第一个是最高阶项拟合系数，次之次高阶 print("数据类型",type(f1))
    f2 = np.polyfit(B, v_2w, 1)

    eff_field = -0.002 * f2[0]/f1[0] # v_2w的单位是uV，1w的单位是mV，二次的相当于数值增大了1000倍因此除以1000
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
def plot_field(fit_result,out_path):
    current, eff_field = [0,], [0,]
    for i in range(len(fit_result) - 1):
        current.append(fit_result[i]['current'])
        eff_field.append(fit_result[i]['fit_value'])
    plot1 = plt.plot(current, eff_field, 's', label="eff_field_DL")

    plt.xlabel('I')
    plt.ylabel("B_DL" )
    plt.legend(loc=4)  # 指定Legend的位置在右下角
    plt.title('B_DL vs I')
    figure = plt.gcf()  # 获取当前图像
    file_name = 'Mz ='+fit_result[0]['Mz']+'result'
    figure.savefig(out_path + file_name + '.png')
    figure.clear()  # 释放内存,没有的话 会把不同曲线画在一个图里，越到后面越多曲线


def caculate_DL(fit_result):
    delta_Bx=[] #计算结果
    for i in range(len(fit_result)-1):
        if fit_result[i]['current'] == fit_result[i+1]['current'] and fit_result[i]['Mz'] == fit_result[i+1]['Mz']:
            if fit_result[i]['harm_order']=='1' and fit_result[i+1]['harm_order']=='2':
                delta_Bx_1 = 0- 2 * fit_result[i+1]['fit_value']/fit_result[i]['fit_value']
                delta_Bx.append({'current':fit_result[i]['current'],
                                 'Mz':fit_result[i]['Mz'],
                                 'deltaB_x':delta_Bx_1})
    return delta_Bx

            
            
