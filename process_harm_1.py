

import operator
import utils
import pprint
file_position = 'D:\\Code\\test_data\\A0Vy'
file_type = '.dat'
B_max = 1000  # 设定小场范围
fit_result = []  # 单个数据文件的拟合结果，包括测试条件和对应谐波的一阶导数或者二阶倒数

file_list, out_path = utils.get_file_list(file_position)

for file_name in file_list:
    test_condition = utils.get_test_condition_1(file_name)  # 获取测试条件
    data = utils.load_data(file_position, file_name)  # 从。dat文件载入数据并给数据排序

    #pos = #utils.find_first_last(data, B_max)  # 找到小场范围

    utils.fit_and_plot_1(file_name, data, B_max, fit_result, out_path, test_condition)
    fit_result.append(test_condition)


pprint.pprint(fit_result)
