import os
def generate_kpath_files(file_path):
    # 读取 IBZKPT.txt 文件内容
    file_count = 1
    with open(file_path +"\IBZKPT.txt", "r") as ibzkpt_file:
        ibzkpt_lines = ibzkpt_file.readlines()

    # 跳过前四行，从第五行开始处理
    ibzkpt_data = ibzkpt_lines[3:]

    # 读取 KPATH.txt 文件作为模板
    with open(file_path + "\KPATH.txt", "r") as kpath_file:
        template_lines = kpath_file.readlines()

    for line_data in ibzkpt_data:
        # 分割一行数据，获取第一列和第二列数值以及最后一列的权重
        kpoint, weight = line_data.split()[:2], int(line_data.split()[-1])

        # 创建新文件并写入模板内容
        out_path = os.path.join(file_path, f"Line_{file_count}_{weight}.txt")
        with open(out_path, "w") as output_file:
            for index, line in enumerate(template_lines):
                # 直接替换第五行和第六行的坐标值
                if index == 4 or index == 5:
                    line = f"{kpoint[0]} {kpoint[1]}   {line[33:]}"
                output_file.write(line)
        file_count += 1
file_path = r"D:\Users\Desktop\计算\GaSe"
generate_kpath_files(file_path)