import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# --- 1. 参数配置 ---

# 数据文件夹
DATA_FOLDER = r'G:\PycharmProject\deepmatcher-master\exp_datasets\1-amazon_google'

# 确保输出目录存在
os.makedirs(DATA_FOLDER, exist_ok=True)

# 输入文件路径
TABLE_A_PATH = os.path.join(DATA_FOLDER, 'tableA.csv')
TABLE_B_PATH = os.path.join(DATA_FOLDER, 'tableB.csv')
MATCHES_PATH = os.path.join(DATA_FOLDER, 'matches.txt')

# 输出文件路径
TRAIN_PATH = os.path.join(DATA_FOLDER, 'train.csv')
VALID_PATH = os.path.join(DATA_FOLDER, 'valid.csv')
TEST_PATH = os.path.join(DATA_FOLDER, 'test.csv')
# 新增的未标记数据文件路径
UNLABELED_PATH = os.path.join(DATA_FOLDER, 'unlabeled.csv')

# 负样本比例
NEGATIVE_SAMPLING_RATIO = 2.0

# --- 核心改动：调整数据集分割比例以容纳未标记数据集 ---
# 新的数据集分割比例
TOTAL_LABELED_SIZE = 0.85 # 85% 用于带标签的数据 (train + valid + test)
UNLABELED_SIZE = 0.15      # 15% 用于未标记数据

# 在带标签的数据中，按比例分配 train, valid, test
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# 随机种子
RANDOM_STATE = 42

# --- 2. 数据加载与合并 ---
print(f"开始从文件夹 '{DATA_FOLDER}' 加载数据...")
try:
    table_a = pd.read_csv(TABLE_A_PATH)
    table_b = pd.read_csv(TABLE_B_PATH)

    # *** 核心改动：立即将两个表上下拼接成一个主表 ***
    combined_table = pd.concat([table_a, table_b], ignore_index=True)

    # 为主表创建 'id' 列，格式为 "idx_行号"
    combined_table['id'] = 'idx_' + combined_table.index.astype(str)

    # 将 id 列设为索引，方便快速查找
    combined_table.set_index('id', inplace=True)

    # 加载正样本匹配对
    with open(MATCHES_PATH, 'r') as f:
        matches = [line.strip().split(',') for line in f]

    print(f"加载成功：主表共有 {len(combined_table)} 条记录。")
    print(f"找到了 {len(matches)} 对正样本。")

except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}。请确保 '{DATA_FOLDER}' 文件夹和您的脚本在同一个目录下。")
    exit()

# --- 3. 生成正样本 ---
print("\n正在生成正样本...")
positive_samples = []
# 使用 set 存储规范化的 ID 对 (min_id, max_id) 以便快速查找
positive_pairs = set()

for id_1, id_2 in matches:
    try:
        # 从主表中获取两行数据
        row_1 = combined_table.loc[id_1]
        row_2 = combined_table.loc[id_2]

        # 横向拼接两行，并重命名列以作区分（使用 _1 和 _2 后缀）
        combined_row = {f"left_{col}": val for col, val in row_1.items()}
        combined_row.update({f"right_{col}": val for col, val in row_2.items()})

        combined_row['label'] = 1

        positive_samples.append(combined_row)

        # 存储规范化的 ID 对，方便后续负采样时检查
        idx_1_num = int(id_1.split('_')[1])
        idx_2_num = int(id_2.split('_')[1])
        positive_pairs.add((min(idx_1_num, idx_2_num), max(idx_1_num, idx_2_num)))

    except KeyError as e:
        print(f"警告：在主表中找不到 ID: {e}。已跳过该匹配对。")

positive_df = pd.DataFrame(positive_samples)
print(f"成功生成 {len(positive_df)} 条正样本。")

# --- 4. 生成负样本 ---
print("\n正在生成负样本...")
num_positive = len(positive_df)
num_negative_to_generate = int(num_positive * NEGATIVE_SAMPLING_RATIO)

negative_samples = []
generated_negative_pairs = set()
all_indices = list(range(len(combined_table)))  # 获取主表的所有数字索引
rng = np.random.default_rng(RANDOM_STATE)

while len(negative_samples) < num_negative_to_generate:
    # 从主表中随机抽取两个 *不同* 的索引
    rand_idx_1, rand_idx_2 = rng.choice(all_indices, size=2, replace=False)

    # 检查该对（或其反向对）是否为正样本或已生成过
    pair_to_check = (min(rand_idx_1, rand_idx_2), max(rand_idx_1, rand_idx_2))

    if pair_to_check not in positive_pairs and pair_to_check not in generated_negative_pairs:
        generated_negative_pairs.add(pair_to_check)

        # 通过数字索引获取行数据
        row_1 = combined_table.iloc[rand_idx_1]
        row_2 = combined_table.iloc[rand_idx_2]
        id_1, id_2 = row_1.name, row_2.name  # .name 属性是该行的索引标签 (e.g., 'idx_123')

        combined_row = {f"left_{col}": val for col, val in row_1.items()}
        combined_row.update({f"right_{col}": val for col, val in row_2.items()})

        combined_row['label'] = 0

        negative_samples.append(combined_row)

negative_df = pd.DataFrame(negative_samples)
print(f"成功生成 {len(negative_df)} 条负样本。")

# --- 5. 合并、创建主ID、打乱与分割 ---
print("\n正在合并与分割数据集...")

# 将正负样本合并成一个完整的带标签数据集
full_labeled_df = pd.concat([positive_df, negative_df], ignore_index=True)

# 随机打乱整个数据集
full_labeled_df = full_labeled_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# --- 核心改动：首先将数据分为带标签和未标记两部分 ---
# 确保分割比例之和为1
if not np.isclose(TOTAL_LABELED_SIZE + UNLABELED_SIZE, 1.0):
    print("错误：带标签数据集和未标记数据集的比例之和不等于1。")
    exit()

# 使用 train_test_split 将数据分成带标签和未标记两部分
labeled_df, unlabeled_df = train_test_split(
    full_labeled_df,
    train_size=TOTAL_LABELED_SIZE,
    random_state=RANDOM_STATE,
    stratify=full_labeled_df['label']
)

# --- 在带标签数据中继续分割训练、验证和测试集 ---
# 确保分割比例之和为1
if not np.isclose(TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE, 1.0):
    print("错误：训练集、验证集和测试集的比例之和不等于1。")
    exit()

# 按照新的比例分割带标签数据集
train_df, temp_df = train_test_split(
    labeled_df,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_STATE,
    stratify=labeled_df['label']
)
validation_proportion = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)
valid_df, test_df = train_test_split(
    temp_df,
    train_size=validation_proportion,
    random_state=RANDOM_STATE,
    stratify=temp_df['label']
)

# --- 为所有数据集添加 'id' 列 ---
for df in [train_df, valid_df, test_df, unlabeled_df]:
    df.reset_index(inplace=True, drop=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id'}, inplace=True)
    cols_to_move = ['id', 'label'] if 'label' in df.columns else ['id']
    df_new_order = cols_to_move + [col for col in df.columns if col not in cols_to_move]
    df = df[df_new_order]

print("数据集分割完成：")
print(f" - 训练集 (train):    {len(train_df)} 条")
print(f" - 验证集 (valid):    {len(valid_df)} 条")
print(f" - 测试集 (test):     {len(test_df)} 条")
print(f" - 未标记集 (unlabeled): {len(unlabeled_df)} 条")

# --- 6. 保存文件 ---
print(f"\n正在向文件夹 '{DATA_FOLDER}' 保存文件...")
try:
    train_df.to_csv(TRAIN_PATH, index=False)
    valid_df.to_csv(VALID_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    # 核心改动：保存未标记数据，并移除 'label' 列
    unlabeled_df.drop(columns=['label'], inplace=True)
    unlabeled_df.to_csv(UNLABELED_PATH, index=False)
    print(f"文件已成功保存到 '{DATA_FOLDER}' 文件夹中。")
except Exception as e:
    print(f"保存文件时出错: {e}")

print("\n处理完成！")