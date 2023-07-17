"""
面外体系小场测量二次谐波
分两次测量一次和二次的数据，放在两个文件夹

"""

import operator
import utils
import pprint
file_position = 'D:\\ptru_gradient\\device 3\\harmonic\\By'
file_type = '.dat'
B_max = 1000  # 设定小场范围
fit_result = []  # 单个数据文件的拟合结果，包括测试条件和对应谐波的一阶导数或者二阶倒数

file_list, out_path = utils.get_file_list(file_position)

for file_name in file_list:
    test_condition = utils.get_test_condition(file_name)  # 获取测试条件
    data = utils.load_data(file_position, file_name)  # 从。dat文件载入数据并给数据排序
    utils.fit_and_plot(file_name, data, B_max, fit_result, out_path, test_condition)

fit_result = sorted(fit_result, key=operator.itemgetter("current", 'Mz', 'harm_order'))  # 用operator进行多级排序

pprint.pprint(fit_result)



delta_Bx = utils.caculate_DL(fit_result)

utils.plot_field(delta_Bx,out_path)

pprint.pprint(delta_Bx)
