import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import interpolate
# 加载数据
filename = r"D:\Users\Desktop\计算\FeGeTe\最新\TDOS.dat"# 替换为你的文件名
'''
data = np.loadtxt(filename,skiprows=1)
energy = data[:, 0]
dos_up = data[:, 1]
dos_down = np.abs(data[:, 2])  # 使用绝对值
'''
eV = 1.6022E-19
h_bar = 1.05257E-34
d = 5E-9
m = 9.1E-31
m_eff = 0.1
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
        dos_up_occ = np.interp(e_range - vb / 2, energy, dos_up, left=0, right=0)
        dos_up_vac = np.interp(e_range + vb / 2, energy, dos_up, left=0, right=0)
        dos_down_occ = np.interp(e_range - vb / 2, energy, dos_down, left=0, right=0)
        dos_down_vac = np.interp(e_range + vb / 2, energy, dos_down, left=0, right=0)

        # 计算平行态电导, 能量窗口内能量高低的隧穿概率认为一致
        conductance_p[i] = simps(dos_up_occ * dos_up_vac + dos_down_occ * dos_down_vac, e_range)
        # cp = np.sum(dos_up_low * dos_up_high + dos_down_low * dos_down_high)
        # 计算反平行态电导
        conductance_ap[i] = simps(dos_up_occ * dos_down_vac, e_range) + simps(dos_down_occ * dos_up_vac, e_range)
        # cap = 2*sum(dos_up_low * dos_down_high) 这么算不对称
        i+=1
    return conductance_p, conductance_ap

#conductance_p, conductance_ap = cal_conduct(vb_range,energy, dos_up, dos_down)
# 两点不够精确，m_eff和psi_0对于每一个kx，ky都不一样，不能用一样的值，继续修改
def cal_conduct_acc(vb_range, energy, dos_up, dos_down, psi_0, e_parallel, d):
    conductance_p, conductance_ap = np.zeros(len(vb_range)), np.zeros(len(vb_range))
    i = 0
    for vb in vb_range:
        vb = abs(vb)
        e_min, e_max = -np.abs(vb)/2, np.abs(vb)/2
        # 生成偏压下的能量范围数组
        e_range = energy[(energy >= e_min) & (energy <= e_max)]
    # 检查是否有足够的能量进行积分
        if len(e_range)== 0:
            tmr_values.append(0)
            continue
    # 生成积分使用的DOS值
        dos_up_occ = np.interp(e_range - vb / 2, energy, dos_up, left=0, right=0)
        dos_up_vac = np.interp(e_range + vb / 2, energy, dos_up, left=0, right=0)
        dos_down_occ = np.interp(e_range - vb / 2, energy, dos_down, left=0, right=0)
        dos_down_vac = np.interp(e_range + vb / 2, energy, dos_down, left=0, right=0)
        # 计算平行态电导, 能量窗口内能量高低的隧穿概率认为不同，通过公式计算得到
        # 能量为E ，偏压为eV下的隧穿概率
        occ_energy = e_range - vb / 2
        tp = np.exp(-2 * (5E-9 / h_bar) *np.sqrt(2*m*m_eff*eV*(psi_0-occ_energy-vb/2+e_parallel)))
        conductance_p[i] = simps(dos_up_occ*tp*dos_up_vac + dos_down_occ*tp*dos_down_vac, e_range)
        conductance_ap[i] = simps(dos_up_occ *tp* dos_down_vac, e_range) + simps(dos_down_occ *tp* dos_up_vac, e_range)
        i+=1
    return conductance_p, conductance_ap
# 计算不同偏压下的TMR
#condunctance_p, conductance_ap = cal_conduct_acc()
#tmr = np.where(conductance_ap != 0, (conductance_p - conductance_ap) / conductance_ap, 0)
#tmr_values.append(tmr)

def cal_conduct_acc1(vb_range, energy, dos_up, dos_down, psi_0, e_parallel, m_eff,d):
    conductance_p, conductance_ap = np.zeros(len(vb_range)), np.zeros(len(vb_range))
    i = 0
    for vb in vb_range:
        vb = abs(vb)
        e_min, e_max = -np.abs(vb)/2, np.abs(vb)/2
        # 生成偏压下的能量范围数组
        e_range = energy[(energy >= e_min) & (energy <= e_max)]
    # 检查是否有足够的能量进行积分
        if len(e_range)== 0:
            tmr_values.append(0)
            continue
    # 生成积分使用的DOS值
        dos_up_occ = np.interp(e_range - vb / 2, energy, dos_up, left=0, right=0)
        dos_up_vac = np.interp(e_range + vb / 2, energy, dos_up, left=0, right=0)
        dos_down_occ = np.interp(e_range - vb / 2, energy, dos_down, left=0, right=0)
        dos_down_vac = np.interp(e_range + vb / 2, energy, dos_down, left=0, right=0)
        # 计算平行态电导, 能量窗口内能量高低的隧穿概率认为不同，通过公式计算得到
        # 能量为E ，偏压为eV下的隧穿概率
        occ_energy = e_range - vb / 2
        tp = np.exp(-2 * (5E-9 / h_bar) *np.sqrt(2*m*m_eff*eV*(psi_0-occ_energy-vb/2+e_parallel)))
        conductance_p[i] = simps(dos_up_occ*tp*dos_up_vac + dos_down_occ*tp*dos_down_vac, e_range)
        conductance_ap[i] = simps(dos_up_occ *tp* dos_down_vac, e_range) + simps(dos_down_occ *tp* dos_up_vac, e_range)
        i+=1
    return conductance_p, conductance_ap



