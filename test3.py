import matplotlib.pyplot as plt

"""
探针台1005测量TMR 数据
找到反转点，磁阻归一化，变温的TMR
"""
import os
import re
import pprint
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.signal as sgn
plt.rcParams['font.sans-serif'] = ['SimHei']	# 显示中文 非衬线字体为黑体
plt.rcParams['axes.unicode_minus'] = False	#显示负号

file_position = r'D:\Test_data\2D MTJ\7.19 fget sns2\新建文件夹\2mv'

file_type = '.txt'
outpath = r'D:\Test_data\2D MTJ\7.19 fget sns2\新建文件夹\2mv\out'
fit_result = []  # 单个数据文件的拟合结果，包括测试条件和对应谐波的一阶导数或者二阶倒数

file_list, out_path = utils.get_file_list(file_position)

# 提取测试条件，并对测试条件排序
def parse_conditions(condition):
    pattern = r'(-?\d+\.?\d*)t\s+(down|up)\s+(-?\d+\.?\d*)mv\s+(-?\d+\.?\d*)k'
    matches = re.findall(pattern, condition)
    if matches:
        magnetic_field, direction, voltage, temperature = matches[0]
        test_condition={'magnetic_field': float(magnetic_field),
                       'direction': -1 if direction == 'down' else 1,
                       'voltage': float(voltage),
                       'temperature': int(temperature)}
        return(test_condition)
    else:
        print(f"No match found for condition: {condition}")
        return None


#pprint.pprint(file_list)
new_list = []

file_list = [file_name for file_name in file_list if parse_conditions(file_name)!=None]#正常倒序删除会漏掉元素，一种方法是倒序索引，跟utils中一样，另一种filter(fun,iterable_obj),或者列表解析b=[i for i in a if a>c]

def get_sort_key(file_name):
    test_condition = parse_conditions(file_name)
    voltage,temperature,direction,field= test_condition['voltage'],test_condition['temperature'],['direction'],test_condition['magnetic_field']
    return(voltage,temperature,direction,field)

file_list = sorted(file_list,key=get_sort_key)
pprint.pp(file_list)

fignum =0
for file in file_list:

    if parse_conditions(file)['voltage'] == 2.0:
        fignum +=1
plt.figure(figsize=(20,5*fignum))
i=0
for file in file_list:

    if parse_conditions(file)['voltage']== 2.0:
        i=i+1
        data_name = file_position+ '\\'+file
        data = np.loadtxt(data_name,dtype = float,comments='#',unpack=False)
        plt.subplot(fignum,1,i)
        plt.plot(data[:,0],data[:,2],'b-',label='T:'+str(parse_conditions(file)['temperature']))
        plt.legend( )
plt.savefig(out_path+'com.png')