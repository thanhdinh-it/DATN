import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Force UTF-8 trên Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_PATH = 'data/processed/BCTC_Retail_Industry_2020_2025.xlsx'
OUTPUT_DIR = './output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

CORE_COLS = [
    'Tong_Tai_San', 'Tai_San_Ngan_Han', 'Hang_Ton_Kho', 
    'Phai_Thu_KH', 'Tong_No', 'Doanh_Thu_Thuan', 
    'Gia_Von', 'Loi_Nhuan_Sau_Thue', 'LCTT_HDKD'
]
VIETNAMESE_LABELS = {
    'Tong_Tai_San': 'Tong Tai San',
    'Tai_San_Ngan_Han': 'Tai San Ngan Han',
    'Hang_Ton_Kho': 'Hang Ton Kho',
    'Phai_Thu_KH': 'Phai Thu Khach Hang',
    'Tong_No': 'Tong No',
    'Doanh_Thu_Thuan': 'Doanh Thu Thuan',
    'Gia_Von': 'Gia Von Hang Ban',
    'Loi_Nhuan_Sau_Thue': 'LN Sau Thue',
    'LCTT_HDKD': 'LCTT (HDKD)'
}

df = pd.read_excel(DATA_PATH, engine='openpyxl')
df_core = df[[c for c in CORE_COLS if c in df.columns]].copy()
for col in df_core.columns:
    df_core[col] = pd.to_numeric(df_core[col], errors='coerce') / 1e9 # Ty VND

print("=== SO LUONG NGOAI LAI (IQR) ===")
for col in df_core.columns:
    Q1 = df_core[col].quantile(0.25)
    Q3 = df_core[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df_core[(df_core[col] < lower) | (df_core[col] > upper)][col].count()
    print(f"{col}: {outliers} ({outliers/len(df)*100:.1f}%)")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(df_core.columns):
    sns.boxplot(y=df_core[col], ax=axes[i], color='#3498db', flierprops={"marker": "x", "color": "red"})
    axes[i].set_title(VIETNAMESE_LABELS.get(col, col), fontsize=12, fontweight='bold')
    axes[i].set_ylabel("Gia tri (Ty VND)")

fig.suptitle('Bieu do Boxplot: Nhan dien Ngoai lai tren 9 Tru cot Tai chinh (Ty VND)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eda_outliers_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Xong ve bieu do!")
