# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import safe_divide, separator

def preprocess_features(df_valid, imbalance_pct, SPW):
    """Tính toán 12 tỷ số tài chính, winsorize, train_test_split và StandardScaler."""
    separator("BƯỚC 4: FEATURE ENGINEERING – 12 TỶ SỐ TÀI CHÍNH")
    
    d = df_valid
    FEATURE_NAMES = [
        'X_ROA', 'X_NPM', 'X_Debt_Ratio', 'X_Current_Ratio',
        'X_Asset_Turnover', 'X_Inv_Turnover', 'X_Quick_Ratio',
        'X_Operating_Margin', 'X_Interest_Coverage', 'X_AR_Turnover',
        'X_CFO_to_Debt', 'X_Asset_Structure'
    ]

    df_valid['X_ROA']              = safe_divide(d['Loi_Nhuan_Sau_Thue'],  d['Tong_Tai_San'],    0.0)
    df_valid['X_NPM']              = safe_divide(d['Loi_Nhuan_Sau_Thue'],  d['Doanh_Thu_Thuan'], 0.0)
    df_valid['X_Debt_Ratio']       = safe_divide(d['Tong_No'],             d['Tong_Tai_San'],    0.0)
    df_valid['X_Current_Ratio']    = safe_divide(d['Tai_San_Ngan_Han'],    d['No_Ngan_Han'],     0.0)
    df_valid['X_Asset_Turnover']   = safe_divide(d['Doanh_Thu_Thuan'],     d['Tong_Tai_San'],    0.0)
    df_valid['X_Inv_Turnover']     = safe_divide(d['Gia_Von'].abs(),        d['Hang_Ton_Kho'],    0.0)
    df_valid['X_Quick_Ratio']      = safe_divide(d['Tai_San_Ngan_Han'] - d['Hang_Ton_Kho'],
                                                  d['No_Ngan_Han'], 0.0)
    df_valid['X_Operating_Margin'] = safe_divide(
        d['Loi_Nhuan_Gop'] - d['Chi_Phi_Ban_Hang'].abs() - d['Chi_Phi_QLDN'].abs(),
        d['Doanh_Thu_Thuan'], 0.0)
    df_valid['X_Interest_Coverage']= safe_divide(d['Loi_Nhuan_Truoc_Thue'],
                                                  d['Chi_Phi_Lai_Vay'].abs(), 0.0)
    df_valid['X_AR_Turnover']      = safe_divide(d['Doanh_Thu_Thuan'], d['Phai_Thu_KH'],   0.0)
    df_valid['X_CFO_to_Debt']      = safe_divide(d['LCTT_HDKD'],       d['Tong_No'],       0.0)
    df_valid['X_Asset_Structure']  = safe_divide(d['Tai_San_Ngan_Han'],d['Tong_Tai_San'],  0.0)

    # Winsorize 1%–99%
    for col in FEATURE_NAMES:
        lo, hi = df_valid[col].quantile(0.01), df_valid[col].quantile(0.99)
        df_valid[col] = df_valid[col].clip(lo, hi)

    # ---------------------------------------------------------
    # XỬ LÝ ĐA CỘNG TUYẾN (MULTICOLLINEARITY)
    # ---------------------------------------------------------
    print("[4.1] Đang kiểm tra Đa cộng tuyến (Pearson Correlation threshold = 0.8)...")
    corr_matrix = df_valid[FEATURE_NAMES].corr().abs()
    
    # Tìm các cặp tương quan cao
    drop_features = set()
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            if corr_matrix.iloc[i, j] > 0.8:
                f1 = FEATURE_NAMES[i]
                f2 = FEATURE_NAMES[j]
                print(f"      ! Cảnh báo: '{f1}' và '{f2}' có tương quan rất cao (r = {corr_matrix.iloc[i,j]:.3f})")
                # Ưu tiên drop biến đằng sau (f2) để giữ lại biến cơ sở
                drop_features.add(f2)
                
    if drop_features:
        print(f"[4.2] Tiến hành loại bỏ {len(drop_features)} tỷ số tài chính bị trùng lặp thông tin: {drop_features}")
        FEATURE_NAMES = [f for f in FEATURE_NAMES if f not in drop_features]
    else:
        print("[4.2] Không phát hiện cặp biến nào vi phạm đa cộng tuyến.")

    print("\n[4.3] Thống kê nhanh đặc trưng sau khi xử lý:")
    print(df_valid[FEATURE_NAMES].describe().round(3).to_string())

    separator("BƯỚC 5: TÁCH TẬP DỮ LIỆU & CHUẨN HOÁ (không SMOTE)")
    
    X = df_valid[FEATURE_NAMES].values
    y = df_valid['Anomaly_Label'].values
    print(f"[5.1] Tổng mẫu: {len(y)} | Nhãn 0: {(y==0).sum()} | Nhãn 1: {(y==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"[5.2] Train: {len(y_train)} | Test: {len(y_test)}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print("[5.3] Đã chuẩn hóa (StandardScaler).")
    print(f"[5.4] Không áp dụng SMOTE. Mất cân bằng {imbalance_pct:.1f}% — "
          f"xử lý bằng class_weight='balanced' (LR/RF) và scale_pos_weight={SPW} (XGB).")

    return X_train_sc, X_test_sc, y_train, y_test, FEATURE_NAMES, df_valid
