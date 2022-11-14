import operator
import utils
import pprint
file_position = 'D:\\ptru_gradient\\device 2\\harmonic'
file_type = '.dat'
B_max = 1000  # 设定小场范围
fit_result = []  # 单个数据文件的拟合结果，包括测试条件和对应谐波的一阶导数或者二阶倒数

file_list, out_path = utils.get_file_list(file_position)

for file_name in file_list:
    test_condition = utils.get_test_condition(file_name)  # 获取测试条件
    data = utils.load_data(file_position, file_name)  # 从。dat文件载入数据并给数据排序

    pos = utils.find_first_last(data, B_max)  # 找到小场范围

    utils.fit_and_plot(file_name, data, pos, fit_result, out_path, test_condition)

fit_result = sorted(fit_result, key=operator.itemgetter("current", 'Mz', 'harm_order'))  # 用operator进行多级排序

delta_Bx = utils.caculate_DL(fit_result)
pprint.pprint(delta_Bx)
