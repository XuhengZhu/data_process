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

file_position = r'D:\Test_data\2D MTJ\7.19 fget sns2'
file_position1 ='D:\\Users\\Desktop\\MOKE\\test'
file_type = '.txt'

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

def tmr_normalized(data):
    y_changes = np.diff(data[:, 2])
    max_index = np.argmax(y_changes)
    max_field = data[max_index,0]
    min_index = np.argmin(y_changes)
    base_r = (np.mean(data[2:max_index-3,2])+np.mean(data[min_index+3:,2]))/2
    data[:,2] = data[:,2]/base_r
    return data,max_field
def find_switch(data):
    y_changes = np.diff(data[:,2])
    max_index = np.argmax()
def draw_fig():
    return 0

union_path = r'D:\Test_data\2D MTJ\7.19 fget sns2\新建文件夹\\'
switch_data_pos = r'D:\Test_data\2D MTJ\7.19 fget sns2\switch point\switch point.txt'
switch_data = np.loadtxt(switch_data_pos, dtype=float,comments ='#',unpack=False)


for i in range(len(file_list)-1):
    test_condition1 = parse_conditions(file_list[i])
    for j in range(i+1,len(file_list)):
        test_condition2= parse_conditions(file_list[j])
         ################### 组合来回的数据 ##########################
        if  test_condition1['voltage'] == test_condition2['voltage'] and test_condition1['magnetic_field']==test_condition2['magnetic_field'] and test_condition1['temperature']==test_condition2['temperature']:
            data_name1 =file_position +'\\'+ file_list[i]

            data_name2 = file_position + '\\' + file_list[j]
            data1 = np.loadtxt(data_name1, skiprows= 3,usecols=[1,6], dtype=float,comments ='#',unpack=False)


            new_column1 = test_condition1['direction'] * (
                    0 - test_condition1['magnetic_field'] + (data1[:, 0] - 8) / (0.450 * 320))

            data1 = np.insert(data1,0,new_column1,axis=1) #第一列是磁场，第二列是时间，第三列为隧穿电阻,磁场已经是带有正负号了

            data1,max_field1 = tmr_normalized(data1)
           # print(data1)
            ##频谱分析
            fs = 1 #采样频率
            N = len(data1[:,2])
            yf = fft(data1[:,2])
            xf = fftfreq(N,1/fs)[:N//2]
            b,a = sgn.butter(N=8,Wn = 0.25,btype='lowpass',analog=False,output='ba')
            n=len(data1[:,2])


           # plt.plot(xf, (2/N) * np.abs(yf[0:N//2]))
           #plt.xlabel('Frequency(B:Hx)'); plt.ylabel('Amplitude(Ohm)')
           # plt.grid(); plt.show()
            #滤波，直接不要低频噪声然后逆变换

            yf[-1] =0
            yf[-2]=0
            data1[:,2]=np.fft.ifft(yf)#只取了一半不知道对不对，还有上面的对称的存在2/N是不是多乘以了个2 ，逆变换出来是复数，这种滤波方法不好
            bb=0

            for k in range(0,35):
                if switch_data[k, 0] == test_condition1['temperature'] and switch_data[k, 1] == test_condition1[
                    'direction'] and switch_data[k, 2] == test_condition2['voltage']:
                    new_column1 = new_column1 + switch_data[k, 3] - tmr_normalized(data1)[1]
                    data1[:,0] = new_column1
                    bb=1
            data2 = np.loadtxt(data_name2, skiprows=3, usecols=[1, 6], dtype=float, comments='#',
                               unpack=False)
            new_column2 = test_condition2['direction'] * (
                    0 - test_condition2['magnetic_field'] + (data2[:, 0] - 8) / (0.450 * 320))
            data2 =np.insert(data2,0,new_column2,axis=1)#第一列是磁场，第二列是时间，第三列为隧穿电阻
            data2,max_field2 = tmr_normalized(data2)

            ##频谱分析
            fs = 1  # 采样频率
            N = len(data2[:, 2])
            yf = fft(data2[:, 2])
            xf = fftfreq(N, 1 / fs)[:N // 2]
           # plt.plot(xf, (2 / N) * np.abs(yf[0:N // 2]))
           # plt.xlabel('Frequency(B:Hx)');
           # plt.ylabel('Amplitude(Ohm)')
           # plt.grid();
           # plt.show()
            # 滤波，直接不要低频噪声然后逆变换

            yf[-1] = 0
            yf[-2] = 0
            data2[:, 2] = np.fft.ifft(yf)
            bb = 0


            for k in range(36):
                if switch_data[k, 0] == test_condition1['temperature'] and switch_data[k, 1] == test_condition2[
                    'direction'] and switch_data[k, 2] == test_condition2['voltage']:
                    new_column2 = new_column2 + switch_data[k, 3] - tmr_normalized(data2)[1]
                    data2[:,0] = new_column2

            data = np.append(data1, data2, axis=0)  # 组合
            if bb==0:
                zero_field = (max_field1+max_field2)/2
                data[:,0] = data[:,0]-zero_field
            #重定零点


            union =union_path +file_list[i]
            np.savetxt(union,data)

            plt.plot(data[:,0],data[:,2],label='T '+str(test_condition2['temperature'])+'K'+file_list[i])
            plt.xlabel('B(T)'),plt.ylabel('TMR')
            plt.legend()

            plt.savefig(out_path + file_list[i][:-3] + '.png', format='png')
            #plt.show()
            plt.close()
            print(max_field1,max_field2)

file_potion =union_path
file_list, out_path = utils.get_file_list(file_position)
