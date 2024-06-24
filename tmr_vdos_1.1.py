import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import interpolate
##作为早期版本的备份，不会报错无法判断相除的地方的警告
eV = 1.6022E-19
h_bar = 1.05257E-34
d = 5E-9
m = 9.1E-31
m_eff = 0.1
# 设置偏压范围
vb_range = np.linspace(-1.5, 1.5, 1000)  # 从-1.5到1.5的偏压,间隔0.005


# 加载数据
#filename = r"D:\Users\Desktop\计算\FeGeTe\最新\TDOS.dat"# 替换为你的文件名
filename = r"D:\Users\Desktop\计算\FeGaTe\47 47 17\dos_eff_FGaT.txt"
data = np.loadtxt(filename,skiprows=1)
energy = data[:, 0]
dos_up = data[:, 1]
dos_down = np.abs(data[:, 2])  # 使用绝对值

# 设置偏压范围
vb_range = np.linspace(-1.5, 1.5, 1000)  # 从-1.5到1.5的偏压,间隔0.005

# 存储TMR结果
tmr_values = []

# 计算不同偏压下的TMR
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


    # 计算平行态电导
    cp = simps(dos_up_low * dos_up_high + dos_down_low * dos_down_high, e_range)
    #cp = np.sum(dos_up_low * dos_up_high + dos_down_low * dos_down_high)
    # 计算反平行态电导
    cap = simps(dos_up_low * dos_down_high, e_range)+simps(dos_up_high * dos_down_low,e_range)
    #cap = 2*sum(dos_up_low * dos_down_high)
    # 防止分母为零
    tmr = (cp - cap) / cap if cap != 0 else 0
    tmr_values.append(tmr)


# 绘制图表
plt.figure(figsize=(8, 6))
plt.plot(vb_range, tmr_values, label='TMR vs Vb')
plt.xlabel('Bias Voltage (V)')
plt.ylabel('TMR')
plt.title('Tunneling Magnetoresistance vs Bias Voltage')
plt.grid(True)
plt.legend()

#data = np.insert(vb_range, 0, tmr_values, axis=1)  # a  xis =1 插入列，axis=0，插入行
data = np.transpose(np.vstack((vb_range,tmr_values)))
#data = np.append(vb_range,tmr_values,axis=0)
# 保存图表
path =r"D:\Users\Desktop\计算\FeGeTe\最新"
plt.savefig(path + r'\tmr_vs_bias_voltage.png')

np.savetxt(path + r'\tmr_vs_bias_voltage.txt',data)

# 显示图表
plt.show()
