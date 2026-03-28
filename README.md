# SPH6004_AY2526_Group_6

利用 MIMIC-IV 数据集（包括静态、时序和文本特征）构建预测模型，估算患者在 ICU 的住院时长（Time-to-discharge）。

## 🚀 快速开始

### 1. 环境准备

你可以选择使用 **Conda** 或 **uv** 来管理你的 Python 环境（==Python Version 3.11==）。

#### 选项 A: 使用 Conda
```bash
# 创建并激活环境
conda create -n sph6004_env python=3.10
conda activate sph6004_env
# 安装依赖
pip install -r requirements.txt
```
选项 B: 使用 uv (推荐，速度更快)
```
# 创建虚拟环境并同步依赖
uv venv -p 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```



### 2. 数据准备与快速预览

由于原始 MIMIC-IV 数据集非常庞大，为了方便快速调试代码和查看特征，提供了一个采样脚本。

**操作步骤：**
1. 确保原始数据存放在 `data/origin/Assignment2_mimic_dataset/` 目录下。
2. 运行脚本，提取每个文件的前 1000 行：
   ```bash
   python data/quick_viewer.py
   ```
3. 运行结束后，采样后的轻量级文件将生成在 `data/processed/` 目录中。
