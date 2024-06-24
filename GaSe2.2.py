# 双程的TMR，来回在一个文件中直接处理
import os
import re
import pprint
from scipy import interpolate
from scipy.optimize import minimize
import utils
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']	# 显示中文 非衬线字体为黑体
plt.rcParams['axes.unicode_minus'] = False	#显示负号

file_position = r"D:\Users\Desktop\重要同步\Fe5GeTe2 homo\B"

file_type = '.txt'

file_list, out_path = utils.get_file_list(file_position)

for file in file_list:
    data_pos = file_position +'\\'+ file
    data = np.loadtxt(data_pos,skiprows=1,dtype=float,comments='#',unpack=False)
    t_data = data[:,0]

    v=4.8/1520
    t_half = t_data[-1]/2
    T = t_data[-1]
    index_half =int(len(t_data)/2)
    B_up = (-1*T/4+t_data[0:index_half:1])*v
    B_down = ((3/4)*T-t_data[index_half:len(t_data):1])*v

    R_up = data[0:index_half:1,6]
    R_down = data[index_half-1:-1:1,6]

    R_base = np.mean(R_up[0:20:1]+R_down[0:20:1])/2

    B_data = np.append(B_up,B_down,axis= 0)

    TMR = np.append(R_up/R_base-1,R_down/R_base-1)
    plt.plot(B_up,R_up)
    plt.plot(B_down,R_down)
    figure =plt.gcf()
    #plt.show()
    figure.savefig(out_path+file[:-4]+'.png')
    figure.clear()



    data=np.insert(data,2,B_data,axis=1)   #  a  xis =1 插入列，axis=0，插入行
    data=np.insert(data,-1,TMR,axis=1)

    np.savetxt(out_path+file,data)

    plt.plot(B_data, TMR)

    figure = plt.gcf()
    # plt.show()
    figure.savefig(out_path + file[:-4] + 'tmr.png')
    figure.clear()