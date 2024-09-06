import random

# 读取数据
with open('train.txt', 'r') as file:
    lines = file.readlines()

# 打乱数据顺序
random.shuffle(lines)

# 计算训练集和验证集的分界点
split_index = int(0.9 * len(lines))

# 划分数据集
train_lines = lines[:split_index]
val_lines = lines[split_index:]

# 写入训练集文件
with open('train_s.txt', 'w') as train_file:
    train_file.writelines(train_lines)

# 写入验证集文件
with open('val.txt', 'w') as val_file:
    val_file.writelines(val_lines)

print("数据集划分完成。训练集（train_s.txt）和验证集（val.txt）已生成。")
