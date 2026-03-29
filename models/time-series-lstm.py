"""
ICU Remaining Length of Stay (RLOS) Prediction
使用 MIMIC-IV 时间序列数据 + LSTM 模型
特征：hr, rr（滑动窗口=8，支持 padding + mask）
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 1. 数据加载 & 预处理
# ─────────────────────────────────────────

def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 统一列名（小写，去空格）
    df.columns = df.columns.str.strip().str.lower()

    # 解析时间
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])

    # 只保留需要的列
    df = df[["stay_id", "hour_ts", "hr", "rr"]].copy()

    # ── 异常值处理（生理范围裁剪）
    # hr: 心率 20–300 bpm
    df["hr"]  = df["hr"].clip(lower=20,  upper=300)
    # rr: 呼吸率 4–60 次/分
    df["rr"]  = df["rr"].clip(lower=4,   upper=60)

    # ── 缺失值：用 **全局中位数** 填充（先算，再 fillna）
    hr_median = df["hr"].median()
    rr_median = df["rr"].median()
    df["hr"] = df["hr"].fillna(hr_median)
    df["rr"] = df["rr"].fillna(rr_median)

    print(f"全局中位数 -> hr: {hr_median:.1f}, rr: {rr_median:.1f}")
    print(f"数据形状: {df.shape}，stay_id 数量: {df['stay_id'].nunique()}")

    # 按 stay_id + 时间排序
    df = df.sort_values(["stay_id", "hour_ts"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────
# 2. 构建样本（滑动窗口 + padding + mask）
# ─────────────────────────────────────────

WINDOW = 8   # 过去 8 小时
FEATURES = ["hr", "rr"]

def build_samples(df: pd.DataFrame):
    """
    对每个 stay_id：
      - discharge_time = 最后一行 hour_ts
      - 对每个时间点 t（索引 i，从 0 开始）：
          * 窗口 = [i-7 : i+1]，不足 8 步时在前面 padding
          * RLOS = (discharge_time - t).total_seconds() / 3600  (小时)
          * 只保留 RLOS > 0 的点（t 不是最后时刻）
    返回：
      X     : (N, 8, 2)   float32
      masks : (N, 8)      bool  True=真实数据，False=padding
      y     : (N,)        float32  RLOS（小时）
      sids  : (N,)        stay_id  用于 group split
    """
    X_list, mask_list, y_list, sid_list = [], [], [], []

    feat_arr_map = {}  # stay_id -> numpy (T, 2)

    for sid, grp in df.groupby("stay_id", sort=False):
        grp = grp.sort_values("hour_ts").reset_index(drop=True)
        discharge_time = grp["hour_ts"].iloc[-1]
        feats = grp[FEATURES].values.astype(np.float32)  # (T, 2)
        times = grp["hour_ts"].values                     # (T,) datetime64

        T = len(grp)
        for i in range(T - 1):   # 最后时刻 RLOS=0，跳过
            # RLOS（小时）
            rlos = (discharge_time - grp["hour_ts"].iloc[i]).total_seconds() / 3600.0
            if rlos <= 0:
                continue

            # 窗口起止
            start = max(0, i - WINDOW + 1)
            window_feats = feats[start : i + 1]          # (k, 2), k <= 8
            k = len(window_feats)

            # padding 在前
            pad_len = WINDOW - k
            if pad_len > 0:
                pad = np.zeros((pad_len, len(FEATURES)), dtype=np.float32)
                window_feats = np.concatenate([pad, window_feats], axis=0)  # (8, 2)

            # mask: True=真实，False=padding
            mask = np.array([False] * pad_len + [True] * k, dtype=bool)  # (8,)

            X_list.append(window_feats)
            mask_list.append(mask)
            y_list.append(rlos)
            sid_list.append(sid)

    X     = np.stack(X_list,    axis=0)   # (N, 8, 2)
    masks = np.stack(mask_list, axis=0)   # (N, 8)
    y     = np.array(y_list,    dtype=np.float32)  # (N,)
    sids  = np.array(sid_list)

    print(f"\n样本总数: {len(y)}")
    print(f"X shape: {X.shape}, masks shape: {masks.shape}, y shape: {y.shape}")
    print(f"RLOS 统计 (小时): mean={y.mean():.1f}, median={np.median(y):.1f}, max={y.max():.1f}")
    return X, masks, y, sids


# ─────────────────────────────────────────
# 3. Dataset & DataLoader
# ─────────────────────────────────────────

class RLOSDataset(Dataset):
    def __init__(self, X, masks, y):
        self.X     = torch.tensor(X,     dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        self.y     = torch.tensor(y,     dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.masks[idx], self.y[idx]


# ─────────────────────────────────────────
# 4. LSTM 模型
# ─────────────────────────────────────────

class LSTMForRLOS(nn.Module):
    """
    X → LSTM → h_t（最后真实时间步）→ FC → y
    利用 mask 取每条样本最后一个真实时间步的隐状态
    """
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, mask):
        """
        x    : (B, 8, 2)
        mask : (B, 8)  True=真实步
        """
        out, _ = self.lstm(x)           # (B, 8, hidden)

        # 取每条样本最后一个 True 的时间步
        # mask.long() -> (B, 8)，最后一个 True 的位置
        last_idx = mask.long().cumsum(dim=1).argmax(dim=1)  # (B,)
        # 如果 mask 全 False（不应发生），fallback 到 -1
        # gather
        last_idx_expanded = last_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
        h_t = out.gather(1, last_idx_expanded).squeeze(1)  # (B, hidden)

        return self.fc(h_t).squeeze(1)   # (B,)


# ─────────────────────────────────────────
# 5. 训练 & 评测
# ─────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_b, mask_b, y_b in loader:
        X_b, mask_b, y_b = X_b.to(device), mask_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        pred = model(X_b, mask_b)
        loss = criterion(pred, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds_all, ys_all = [], []
    total_loss = 0.0
    for X_b, mask_b, y_b in loader:
        X_b, mask_b, y_b = X_b.to(device), mask_b.to(device), y_b.to(device)
        pred = model(X_b, mask_b)
        total_loss += criterion(pred, y_b).item() * len(y_b)
        preds_all.append(pred.cpu().numpy())
        ys_all.append(y_b.cpu().numpy())
    preds = np.concatenate(preds_all)
    ys    = np.concatenate(ys_all)
    mae   = mean_absolute_error(ys, preds)
    rmse  = np.sqrt(mean_squared_error(ys, preds))
    return total_loss / len(loader.dataset), mae, rmse, preds, ys


# ─────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────

def main():
    DATA_PATH   = "/home/myCourse/sph6004/SPH6004_AY2526_Group_6/data/origin/Assignment2_mimic_dataset/MIMIC-IV-time_series(Group Assignment).csv"   # ← 修改为你的文件路径
    BATCH_SIZE  = 256
    EPOCHS      = 30
    LR          = 1e-3
    HIDDEN_DIM  = 64
    NUM_LAYERS  = 2
    SEED        = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # ── 加载 & 预处理
    df = load_and_preprocess(DATA_PATH)

    # ── 构建样本
    X, masks, y, sids = build_samples(df)

    # ── 按 stay_id 划分（train 70% / val 15% / test 15%）
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(gss.split(X, y, groups=sids))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(gss2.split(X[temp_idx], y[temp_idx], groups=sids[temp_idx]))
    val_idx  = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    print(f"\n数据划分 -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    print(f"train stay_ids: {len(np.unique(sids[train_idx]))}, "
          f"val: {len(np.unique(sids[val_idx]))}, "
          f"test: {len(np.unique(sids[test_idx]))}")

    # ── Dataset / Loader
    train_ds = RLOSDataset(X[train_idx], masks[train_idx], y[train_idx])
    val_ds   = RLOSDataset(X[val_idx],   masks[val_idx],   y[val_idx])
    test_ds  = RLOSDataset(X[test_idx],  masks[test_idx],  y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── 模型
    model     = LSTMForRLOS(input_dim=len(FEATURES), hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5) #type:ignore

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # ── 训练
    best_val_mae = float("inf")
    best_state   = None

    print(f"{'Epoch':>6} | {'TrainLoss':>10} | {'ValLoss':>9} | {'Val MAE':>8} | {'Val RMSE':>9}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, val_rmse, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>9.4f} | {val_mae:>8.2f}h | {val_rmse:>9.2f}h")

    # ── 测试集评测
    model.load_state_dict(best_state) #type:ignore
    _, test_mae, test_rmse, preds, ys = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 55)
    print("【测试集评测结果】")
    print(f"  MAE  : {test_mae:.2f} 小时")
    print(f"  RMSE : {test_rmse:.2f} 小时")
    print(f"  实际 RLOS 均值: {ys.mean():.2f}h，预测均值: {preds.mean():.2f}h")

    # ── 分桶误差分析（按真实 RLOS 分段）
    print("\n【按 RLOS 区间的 MAE 分析】")
    bins   = [0, 12, 24, 48, 96, float("inf")]
    labels = ["<12h", "12-24h", "24-48h", "48-96h", ">96h"]
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        mask_b = (ys >= lo) & (ys < hi)
        if mask_b.sum() == 0:
            continue
        mae_b = mean_absolute_error(ys[mask_b], preds[mask_b])
        print(f"  {lab:>8}: N={mask_b.sum():>6}, MAE={mae_b:.2f}h")
    print("=" * 55)


if __name__ == "__main__":
    main()