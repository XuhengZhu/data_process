
"""
面内体系
转磁场角度测量二次谐波  ，同时测量一次和二次数据
多个磁场下测量得到 拟合值与电流的关系，再拟合得到这个电流下的有效场

"""

import os

import pprint
import utils

file_position = 'D:\\Test_data\\zhushouguo\\Orbital Hall - 副本\CuO\\2-Cu6Pt2Co3 - 副本'
file_type = '.dat'

fit_result = []  # 单个数据文件的拟合结果，包括测试条件和对应谐波的一阶导数或者二阶倒数

file_list, out_path = utils.get_file_list(file_position)
#   合并文件并且排序
utils.del_file_nouse(file_list)

union_path = file_position + '\\union\\'  # 缝合文件后的位置
if not os.path.exists(union_path):
    os.mkdir(union_path)

AHE_file = ''  # 找到AHEfile，这个部分待编写

#  缝合文件。有两个半圈的转磁场角度的文件
utils.zip_files(file_list,union_path,file_position)
#   重新换到处理后的path，outpath位置不变
file_position = union_path
file_list, a = utils.get_file_list(file_position)
# pprint.pprint(file_list)
V_AHE = 540.4/2  # 以uv计,注意除以2
cu_ahe = 3  # 测量V-AHE所用的电流
Hk =6726

for file_name in file_list:
    test_condition = utils.get_test_condition_rotate(file_name)  # 获取测试条件

    data = utils.load_data_rotate(file_position, file_name)  # 从.dat文件载入数据并给数据排序

    utils.fit_and_plot_rotate(file_name, data, fit_result, out_path, test_condition)  # 画图并处理数据

    fit_result.append(test_condition)

# pprint.pprint(fit_result)

result_reshape = utils.reshape_test_result(fit_result,V_AHE,cu_ahe,Hk)

pprint.pprint(result_reshape)

utils.fit_save_csv(result_reshape, out_path)





