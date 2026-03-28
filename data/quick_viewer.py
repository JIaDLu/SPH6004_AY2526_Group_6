import pandas as pd
import os

# 1. 定义路径
input_dir = "/home/myCourse/sph6004/SPH6004_AY2526_Group_6/data/origin/Assignment2_mimic_dataset"
output_dir = "/home/myCourse/sph6004/SPH6004_AY2526_Group_6/data/processed"

# 如果 processed 文件夹不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 待处理的文件列表
files = [
    "MIMIC-IV-static(Group Assignment).csv",
    "MIMIC-IV-text(Group Assignment).csv",
    "MIMIC-IV-time_series(Group Assignment).csv"
]

# 3. 循环处理：读取前1000行并导出
for file_name in files:
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, f"sample_{file_name}")
    
    print(f"正在处理: {file_name}...")
    
    # 只读取前 1000 行
    df_sample = pd.read_csv(input_path, nrows=1000)
    
    # 导出到 processed 文件夹
    df_sample.to_csv(output_path, index=False)
    print(f"已导出采样文件至: {output_path}")

print("\n所有采样文件处理完毕！可以在 data/processed目录下快速查看。")