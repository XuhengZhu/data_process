import numpy as np
import operator
import utils
import pprint
file_position = r'D:\Users\Desktop\GaSe\Fe3GaTe2-GaSe-Fe3GaTe2\device11-正文\bianwen\新建文件夹'
file_type = '.txt'

fit_result = []  # 单个数据文件的拟合结果，包括测试条件和对应谐波的一阶导数或者二阶倒数

file_list, out_path = utils.get_file_list(file_position)


for file_name in file_list:
    data_name = file_position + "\\" + file_name  # 为了更改起来更方便

    data = np.loadtxt(data_name, skiprows=1,usecols=range(1,6) , dtype=float, comments='#',
                             unpack=False)  # unpack是一行是一行，一列是一列，false会把列变行


    # 去掉大场部分，先排序,这里只做了排序
    data = data[data[:, 0].argsort()]  # 按照第一列（dianya）降序排序
    unique_x = np.unique(data[:,0])
    #创建空列表存储结果
    result=[]
    for x in unique_x:
        y_values = data[data[:,0]== x,1:]# 选择除第一列外的所有列
        average_y = np.mean(y_values,axis=0)
        result.append([x]+list(average_y))
    np.savetxt(file_position+'\sorted'+file_name,result)