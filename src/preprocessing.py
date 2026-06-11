# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from src.utils import safe_divide, separator

def load_data(file_path: str):
    """Nạp dữ liệu từ file Excel và thực hiện dọn dẹp biến cơ bản."""
    separator("BƯỚC 1: NẠP DỮ LIỆU VÀ TIỀN XỬ LÝ")
    print(f"[1.1] Đọc file: {file_path}")
    df = pd.read_excel(file_path, engine='openpyxl')
    print(f"      Kích thước ban đầu: {df.shape}")

    # Ánh xạ tên cột tiếng Việt → alias ngắn
    COL_ALIASES = {
        '7.1. Trong đó: Chi phí lãi vay': 'Chi_Phi_Lai_Vay',
    }
    for raw, alias in COL_ALIASES.items():
        if raw in df.columns and alias not in df.columns:
            df[alias] = df[raw]
            print(f"      [map] '{raw}' → '{alias}'")

    REQUIRED_COLS = [
        'Nam', 'Ten_Cty', 'Tong_Tai_San', 'Tai_San_Ngan_Han', 'Hang_Ton_Kho',
        'Phai_Thu_KH', 'PPE_Rong', 'Khau_Hao_LK', 'Tong_No', 'No_Ngan_Han',
        'No_Dai_Han_Vay', 'Doanh_Thu_Thuan', 'Gia_Von', 'Loi_Nhuan_Gop',
        'Chi_Phi_Ban_Hang', 'Chi_Phi_QLDN', 'Loi_Nhuan_Sau_Thue',
        'LCTT_HDKD', 'Loi_Nhuan_Truoc_Thue', 'Chi_Phi_Lai_Vay'
    ]

    print("[1.2] Kiểm tra và tạo các cột còn thiếu (= 0):")
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = 0
            print(f"      ⚠  Tạo mới cột: {col}")

    for col in REQUIRED_COLS:
        if col != 'Ten_Cty':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print("[1.3] Lọc: chỉ giữ dòng có Doanh_Thu_Thuan > 0")
    df = df[df['Doanh_Thu_Thuan'] > 0].copy()
    print(f"      Kích thước sau lọc: {df.shape}")
    print(f"      Số công ty: {df['Ten_Cty'].nunique()} | Năm: {sorted(df['Nam'].unique())}")
    
    return df


def calculate_mscore_and_label(df):
    """Tính toán 8 biến số Beneish, M-Score, và gán nhãn."""
    separator("BƯỚC 2: PROXY LABELING – BENEISH M-SCORE")
    df = df.sort_values(['Ten_Cty', 'Nam']).reset_index(drop=True)
    grp = df.groupby('Ten_Cty')
    lag = lambda col: grp[col].shift(1)

    print("[2.1] Tính 8 biến Beneish…")

    AR_t, Rev_t = df['Phai_Thu_KH'], df['Doanh_Thu_Thuan']
    AR_l, Rev_l  = lag('Phai_Thu_KH'), lag('Doanh_Thu_Thuan')
    df['DSRI'] = safe_divide(safe_divide(AR_t, Rev_t, np.nan),
                              safe_divide(AR_l, Rev_l, np.nan), fill_nan=1.0).fillna(1.0)

    GM_t = safe_divide(df['Loi_Nhuan_Gop'], Rev_t, np.nan)
    GM_l = safe_divide(lag('Loi_Nhuan_Gop'), lag('Doanh_Thu_Thuan'), np.nan)
    df['GMI'] = safe_divide(GM_l, GM_t, fill_nan=1.0).fillna(1.0)

    CA_t, PPE_t, TA_t = df['Tai_San_Ngan_Han'], df['PPE_Rong'], df['Tong_Tai_San']
    CA_l, PPE_l, TA_l = lag('Tai_San_Ngan_Han'), lag('PPE_Rong'), lag('Tong_Tai_San')
    df['AQI'] = safe_divide(1 - safe_divide(CA_t + PPE_t, TA_t, np.nan),
                             1 - safe_divide(CA_l + PPE_l, TA_l, np.nan), fill_nan=1.0).fillna(1.0)

    df['SGI'] = safe_divide(Rev_t, Rev_l, fill_nan=1.0).fillna(1.0)

    Dep_t, Dep_l = df['Khau_Hao_LK'].abs(), lag('Khau_Hao_LK').abs()
    df['DEPI'] = safe_divide(safe_divide(Dep_l, PPE_l + Dep_l, np.nan),
                              safe_divide(Dep_t, PPE_t + Dep_t, np.nan), fill_nan=1.0).fillna(1.0)

    SGA_t, SGA_l = df['Chi_Phi_Ban_Hang'].abs() + df['Chi_Phi_QLDN'].abs(), lag('Chi_Phi_Ban_Hang').abs() + lag('Chi_Phi_QLDN').abs()
    df['SGAI'] = safe_divide(safe_divide(SGA_t, Rev_t, np.nan),
                              safe_divide(SGA_l, Rev_l, np.nan), fill_nan=1.0).fillna(1.0)

    TL_t, TL_l = df['Tong_No'], lag('Tong_No')
    df['LVGI'] = safe_divide(safe_divide(TL_t, TA_t, np.nan),
                              safe_divide(TL_l, TA_l, np.nan), fill_nan=1.0).fillna(1.0)

    df['TATA'] = safe_divide(df['Loi_Nhuan_Sau_Thue'] - df['LCTT_HDKD'],
                              TA_t, fill_nan=1.0).fillna(1.0)

    # M-Score
    df['M_Score'] = (
        -4.840 + 0.920*df['DSRI'] + 0.528*df['GMI'] + 0.404*df['AQI']
        + 0.892*df['SGI'] + 0.115*df['DEPI'] - 0.172*df['SGAI']
        + 4.679*df['TATA'] - 0.327*df['LVGI']
    )
    df['Anomaly_Label'] = (df['M_Score'] > -2.22).astype(int)
    print(f"[2.2] Tỷ lệ bất thường trước khi drop 2020: {df['Anomaly_Label'].mean()*100:.1f}%")

    df_valid = df[df['Nam'] != 2020].copy().reset_index(drop=True)
    print(f"[2.3] Sau khi bỏ năm 2020: {df_valid.shape}")
    vc = df_valid['Anomaly_Label'].value_counts().sort_index()
    print(f"      Phân phối nhãn:\n{vc.to_string()}")

    n0, n1 = int(vc.get(0, 0)), int(vc.get(1, 0))
    SPW = round(n0 / n1, 4)
    imbalance_pct = abs(n0 - n1) / (n0 + n1) * 100
    print(f"      Mất cân bằng: {imbalance_pct:.1f}%  |  scale_pos_weight = {SPW}")

    return df_valid, imbalance_pct, SPW
