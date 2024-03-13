import re

import utils
import pprint
# 打开文件
file_position =r'D:\Users\Desktop\重要同步\FGaT_GaSe_FGeT\2.2FGaTGaSeFGeT\c\tmr\170k'
dd, oh = utils.get_file_list(file_position)
utils.delet_unit1(file_position)
pprint(dd)