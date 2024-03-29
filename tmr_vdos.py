import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import interpolate
# 加载数据
filename = r"D:\Users\Desktop\计算\FeGaTe\414113\atdos_tmr_w.dat"# 替换为你的文件名
data = np.loadtxt(filename,skiprows=1)
energy = data[:, 0]
dos_up = data[:, 1]
dos_down = np.abs(data[:, 2])  # 使用绝对值

# 设置偏压范围
vb_range = np.linspace(-1.5, 1.5, 1000)  # 从-1.5到1.5的偏压,间隔0.005

# 存储TMR结果
tmr_values = []
def cal_conduct(vb_range,energy, dos_up, dos_down):
    conductance_p = np.zeros(len(vb_range))
    conductance_ap = np.zeros(len(vb_range))
    i = 0
    for vb in vb_range:

        e_min, e_max = -np.abs(vb) / 2, np.abs(vb) / 2
        # 生成偏压下的能量范围数组
        e_range = energy[(energy >= e_min) & (energy <= e_max)]

        # 检查是否有足够的能量点进行积分
        if len(e_range) == 0:
            tmr_values.append(0)  # 如果没有有效范围，TMR设为0
            continue

        # 生成积分使用的DOS值
        dos_up_low = np.interp(e_range - vb / 2, energy, dos_up, left=0, right=0)
        dos_up_high = np.interp(e_range + vb / 2, energy, dos_up, left=0, right=0)
        dos_down_low = np.interp(e_range - vb / 2, energy, dos_down, left=0, right=0)
        dos_down_high = np.interp(e_range + vb / 2, energy, dos_down, left=0, right=0)

        # 计算平行态电导, 能量窗口内能量高低的隧穿概率认为一致
        conductance_p[i] = simps(dos_up_low * dos_up_high + dos_down_low * dos_down_high, e_range)
        # cp = np.sum(dos_up_low * dos_up_high + dos_down_low * dos_down_high)
        # 计算反平行态电导
        conductance_ap[i] = simps(dos_up_low * dos_down_high, e_range) + simps(dos_up_high * dos_down_low, e_range)
        # cap = 2*sum(dos_up_low * dos_down_high) 这么算不对称
        i+=1
    return conductance_p, conductance_ap

conductance_p, conductance_ap = cal_conduct(vb_range,energy, dos_up, dos_down)
# 计算不同偏压下的TMR
tmr = np.where(conductance_ap != 0, (conductance_p - conductance_ap) / conductance_ap, 0)
tmr_values.append(tmr)





