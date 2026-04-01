import pandas as pd
import sys

# ── 1. 文件路径 ──────────────────────────────────────────────────────────────
BASE = "/home/myCourse/sph6004/SPH6004_AY2526_Group_6/data/origin/Assignment2_mimic_dataset"
STATIC_PATH = f"{BASE}/MIMIC-IV-static(Group Assignment).csv"
TEXT_PATH   = f"{BASE}/MIMIC-IV-text(Group Assignment).csv"
TS_PATH     = f"{BASE}/MIMIC-IV-time_series(Group Assignment).csv"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── 2. 读取数据 ───────────────────────────────────────────────────────────────
section("读取数据")
static = pd.read_csv(STATIC_PATH)
text   = pd.read_csv(TEXT_PATH)
ts     = pd.read_csv(TS_PATH)

print(f"Static    shape : {static.shape}")
print(f"Text      shape : {text.shape}")
print(f"TS        shape : {ts.shape}")

# ── 3. Static stay_id 唯一性 ──────────────────────────────────────────────────
section("Step 1 │ Static — stay_id 唯一性检查")
static_ids   = static["stay_id"]
n_total      = len(static_ids)
n_unique     = static_ids.nunique()
n_duplicate  = n_total - n_unique
duplicated   = static_ids[static_ids.duplicated(keep=False)]

print(f"  总行数       : {n_total}")
print(f"  唯一 stay_id : {n_unique}")
print(f"  重复行数     : {n_duplicate}")

if n_duplicate == 0:
    print("  ✅ Static 的 stay_id 完全唯一，可作为主键使用。")
else:
    print("  ❌ Static 的 stay_id 存在重复！重复值如下：")
    print(duplicated.value_counts().to_string())
    print("\n⚠️  由于 static stay_id 不唯一，后续比较结果仅供参考。")

# ── 4. 构建唯一集合 ───────────────────────────────────────────────────────────
static_set = set(static["stay_id"].unique())
text_set   = set(text["stay_id"].unique())
ts_set     = set(ts["stay_id"].unique())

section("Step 2 │ 各表 stay_id 唯一值数量")
print(f"  Static  唯一 stay_id : {len(static_set):>6}")
print(f"  Text    唯一 stay_id : {len(text_set):>6}")
print(f"  TS      唯一 stay_id : {len(ts_set):>6}")

# ── 5. Text vs Static ─────────────────────────────────────────────────────────
section("Step 3 │ Text — 与 Static 的对应关系")
text_only   = text_set - static_set    # 在 text 中但不在 static 中
static_no_text = static_set - text_set # 在 static 中但不在 text 中

print(f"  Text 中有、Static 中无（孤立 stay_id）: {len(text_only)}")
if text_only:
    sample = sorted(list(text_only))[:20]
    print(f"  示例（最多20个）: {sample}")

print(f"  Static 中有、Text 中无（缺失 stay_id）: {len(static_no_text)}")
if static_no_text:
    sample = sorted(list(static_no_text))[:20]
    print(f"  示例（最多20个）: {sample}")

if not text_only and not static_no_text:
    print("  ✅ Text 与 Static 的 stay_id 完全一一对应。")
elif not text_only:
    print("  ✅ Text 中所有 stay_id 均可在 Static 中找到（Text ⊆ Static）。")
else:
    print("  ❌ Text 中存在无法在 Static 中找到的 stay_id！")

# ── 6. Time-Series vs Static ──────────────────────────────────────────────────
section("Step 4 │ Time-Series — 与 Static 的对应关系")
ts_only        = ts_set - static_set
static_no_ts   = static_set - ts_set

print(f"  TS 中有、Static 中无（孤立 stay_id）: {len(ts_only)}")
if ts_only:
    sample = sorted(list(ts_only))[:20]
    print(f"  示例（最多20个）: {sample}")

print(f"  Static 中有、TS 中无（缺失 stay_id）: {len(static_no_ts)}")
if static_no_ts:
    sample = sorted(list(static_no_ts))[:20]
    print(f"  示例（最多20个）: {sample}")

if not ts_only and not static_no_ts:
    print("  ✅ Time-Series 与 Static 的 stay_id 完全一一对应。")
elif not ts_only:
    print("  ✅ Time-Series 中所有 stay_id 均可在 Static 中找到（TS ⊆ Static）。")
else:
    print("  ❌ Time-Series 中存在无法在 Static 中找到的 stay_id！")

# ── 7. 三表交集总结 ───────────────────────────────────────────────────────────
section("Step 5 │ 三表整体汇总")
all_three   = static_set & text_set & ts_set
only_static = static_set - text_set - ts_set
only_text   = text_set   - static_set
only_ts     = ts_set     - static_set
text_ts_no_static = (text_set | ts_set) - static_set

print(f"  三表共有的 stay_id            : {len(all_three)}")
print(f"  仅在 Static 中（Text/TS 都没有）: {len(only_static)}")
print(f"  仅在 Text 中（Static 没有）    : {len(only_text)}")
print(f"  仅在 TS 中（Static 没有）      : {len(only_ts)}")
print(f"  Text 或 TS 中有但 Static 无    : {len(text_ts_no_static)}")

# ── 8. 最终结论 ───────────────────────────────────────────────────────────────
section("最终结论")
all_ok = (n_duplicate == 0) and (not text_only) and (not ts_only)
if all_ok:
    print("  🎉 所有检查通过：")
    print("     • Static stay_id 唯一")
    print("     • Text 与 TS 中所有 stay_id 均能在 Static 中找到")
    print("     • 数据集关联关系完整一致")
else:
    issues = []
    if n_duplicate > 0:
        issues.append(f"Static 存在 {n_duplicate} 行重复 stay_id")
    if text_only:
        issues.append(f"Text 中有 {len(text_only)} 个 stay_id 在 Static 中找不到")
    if ts_only:
        issues.append(f"TS 中有 {len(ts_only)} 个 stay_id 在 Static 中找不到")
    print("  ⚠️  发现以下问题：")
    for i in issues:
        print(f"     • {i}")