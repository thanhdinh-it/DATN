# -*- coding: utf-8 -*-
"""
Script phân tích thống kê mô tả và trực quan hóa phân phối Raw Data
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import Tee logger từ pipeline
from src.utils import setup_logger

# Force UTF-8 trên Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# -- CẤU HÌNH ĐƯỜNG DẪN --
DATA_PATH = 'data/processed/BCTC_Retail_Industry_2020_2025.xlsx'
OUTPUT_DIR = './output'

# 9 cột tài chính cốt lõi
CORE_COLS = [
    'Tong_Tai_San', 'Tai_San_Ngan_Han', 'Hang_Ton_Kho', 
    'Phai_Thu_KH', 'Tong_No', 'Doanh_Thu_Thuan', 
    'Gia_Von', 'Loi_Nhuan_Sau_Thue', 'LCTT_HDKD'
]

# Ánh xạ tên cột tiếng Việt hiển thị trên biểu đồ
VIETNAMESE_LABELS = {
    'Tong_Tai_San': 'Tổng Tài Sản',
    'Tai_San_Ngan_Han': 'Tài Sản Ngắn Hạn',
    'Hang_Ton_Kho': 'Hàng Tồn Kho',
    'Phai_Thu_KH': 'Phải Thu Khách Hàng',
    'Tong_No': 'Tổng Nợ',
    'Doanh_Thu_Thuan': 'Doanh Thu Thuần',
    'Gia_Von': 'Giá Vốn Hàng Bán',
    'Loi_Nhuan_Sau_Thue': 'Lợi Nhuận Sau Thuế',
    'LCTT_HDKD': 'Lưu Chuyển Tiền Thuần (HĐKD)'
}

def format_vietnamese_number(x):
    """Định dạng số nguyên với dấu chấm (.) phân cách hàng nghìn."""
    if pd.isna(x):
        return "NaN"
    return "{:,.0f}".format(x).replace(",", ".")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Kích hoạt logging ra file
    log_file_path = os.path.join(OUTPUT_DIR, 'raw_data_stats_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    # Backup STDOUT
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    
    # Dùng class Tee tương tự pipeline
    from src.utils import Tee
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    try:
        print("="*70)
        print(f" ĐANG ĐỌC FILE: {DATA_PATH}")
        print("="*70)
        
        try:
            df = pd.read_excel(DATA_PATH, engine='openpyxl')
            print(f"Kích thước dữ liệu gốc: {df.shape[0]} dòng, {df.shape[1]} cột.\n")
        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")
            return

        available_cols = [c for c in CORE_COLS if c in df.columns]
        if len(available_cols) < len(CORE_COLS):
            missing = set(CORE_COLS) - set(available_cols)
            print(f"[Cảnh báo] Thiếu các cột sau trong data: {missing}")

        df_core = df[available_cols].copy()
        for c in available_cols:
            df_core[c] = pd.to_numeric(df_core[c], errors='coerce')

        # ==========================================
        # Xuất Thống Kê
        # ==========================================
        desc = df_core.describe().T[['count', 'mean', 'std', 'min', 'max']]
        for col in ['mean', 'std', 'min', 'max']:
            desc[col] = desc[col].apply(format_vietnamese_number)
        desc['count'] = desc['count'].apply(lambda x: format_vietnamese_number(int(x)))

        print("BẢNG THỐNG KÊ MÔ TẢ 9 ĐẶC TRƯNG TÀI CHÍNH CỐT LÕI (RAW DATA):\n")
        header = "| Feature | " + " | ".join(desc.columns) + " |"
        separator_md = "|" + "|".join(["---"] * (len(desc.columns) + 1)) + "|"
        print(header)
        print(separator_md)
        for index, row in desc.iterrows():
            row_str = " | ".join([str(val) for val in row.values])
            print(f"| {index} | {row_str} |")

        print("\n" + "="*70)
        print(" ĐANG VẼ BIỂU ĐỒ (ViolinPlot & ECDF) ...")
        print("="*70)

        sns.set_theme(style="whitegrid")

        # Chuẩn bị dữ liệu vẽ (Chuyển về Tỷ VNĐ và giới hạn outliers 5%-95%)
        df_plot = pd.DataFrame()
        for col in available_cols:
            series = df_core[col].dropna() / 1e9 # Tỷ VNĐ
            
            # Cắt ngoại lai (Winsorization nhẹ) chỉ để hình vẽ không bị bóp méo
            lower_bound = series.quantile(0.05)
            upper_bound = series.quantile(0.95)
            df_plot[col] = series.clip(lower=lower_bound, upper=upper_bound)

        # ==========================================
        # BIỂU ĐỒ 1: VIOLIN PLOT (Thay thế Boxplot)
        # ==========================================
        fig1, axes1 = plt.subplots(3, 3, figsize=(15, 12))
        axes1 = axes1.flatten()

        for i, col in enumerate(available_cols):
            sns.violinplot(x=df_plot[col], ax=axes1[i], color='#9b59b6', inner="quartile")
            axes1[i].set_title(VIETNAMESE_LABELS.get(col, col), fontsize=12, fontweight='bold')
            axes1[i].set_xlabel("Giá trị (Tỷ VNĐ)")
            
        for j in range(len(available_cols), 9):
            fig1.delaxes(axes1[j])
            
        fig1.suptitle('Phân phối Biên độ Mật độ Dữ liệu Bán lẻ (Tỷ VNĐ - đã loại trừ nhóm Outlier 5%)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot1_path = os.path.join(OUTPUT_DIR, 'raw_data_violinplots.png')
        fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # ==========================================
        # BIỂU ĐỒ 2: ECDF (Thay thế Histogram)
        # ==========================================
        fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
        axes2 = axes2.flatten()

        for i, col in enumerate(available_cols):
            sns.ecdfplot(data=df_plot[col], ax=axes2[i], color='#e74c3c', linewidth=2)
            axes2[i].set_title(VIETNAMESE_LABELS.get(col, col), fontsize=12, fontweight='bold')
            axes2[i].set_xlabel("Giá trị (Tỷ VNĐ)")
            axes2[i].set_ylabel("Tỷ lệ phân phối Tích luỹ (ECDF)")

        for j in range(len(available_cols), 9):
            fig2.delaxes(axes2[j])

        fig2.suptitle('Biểu đồ Tích lũy Xác suất ECDF Dữ liệu Bán lẻ (Tỷ VNĐ)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot2_path = os.path.join(OUTPUT_DIR, 'raw_data_ecdf_plots.png')
        fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)

        print(f" [Hoàn thành] Đã lưu {plot1_path}")
        print(f" [Hoàn thành] Đã lưu {plot2_path}")
        print("="*70)
        print(f"\n[LOG] Toàn bộ terminal log đã lưu tại: {log_file_path}")

    finally:
        sys.stdout = getattr(_orig_stdout, 'streams', [sys.stdout])[0] if hasattr(_orig_stdout, 'streams') else _orig_stdout
        sys.stderr = getattr(_orig_stderr, 'streams', [sys.stderr])[0] if hasattr(_orig_stderr, 'streams') else _orig_stderr
        log_file.close()


if __name__ == '__main__':
    main()
