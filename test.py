import re

def parse_current_and_field(string):
    # 定义正则表达式模式
    pattern = r'Hx_([+-]?[\d.]+)_I_([+-]?[\d.]+) (m|u)A'

    # 匹配正则表达式
    matches = re.search(pattern, string)

    if matches is None:
        # 未找到匹配项，返回 None
        pattern = r'Hx_([+-]?[\d.]+)_I_([+-]?[\d]+[d]+[\d]) (m|u)A'
        matches = re.search(pattern, string)

    try:
        # 获取匹配项中的电流和磁场大小
        field = float(matches.group(1).replace('d', '.'))
        current = float(matches.group(2).replace('d', '.'))
        unit = matches.group(3)

        # 转换电流大小为 mA
        if unit == 'u':
            current /= 1000

        # 返回结果
        return {
            'field': field,
            'current': current
        }
    except:
        # 解析出错，返回 None
        return None

# 测试示例字符串
example_string1 = 'MOnly_PtGd_2_Hz_300_Hx_-160_I_1mA.txt'
example_string2 = 'Only_PtGd_2_Hz_300_Hx_-80_I_-1mA.txt'
example_string3 = 'Only_PtGd_2_Hz_300_Hx_-80_I_1mA.txt'
example_string4 = 'Mo_0d5_PtGd_1d5_Hz_200_Hx_20_I_-0d2 uA'

# 解析示例字符串并输出结果
print(parse_current_and_field(example_string1))
print(parse_current_and_field(example_string2))
print(parse_current_and_field(example_string3))
print(parse_current_and_field(example_string4))