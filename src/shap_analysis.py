# -*- coding: utf-8 -*-
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.utils import separator

def run_shap_analysis(model, X_test_sc, feature_names, output_dir='./output'):
    """
    Thực hiện phân tích SHAP cho mô hình (ưu tiên XGBoost) 
    và lưu các biểu đồ vào thư mục ./shap với phong cách chuyên nghiệp.
    """
    shap_dir = './shap'
    os.makedirs(shap_dir, exist_ok=True)
    
    # Thiết lập phong cách đồ thị
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    separator("PHỤ LỤC: PHÂN TÍCH GIẢI THÍCH MÔ HÌNH (SHAP)")
    print(f"[SHAP] Đang khởi tạo Explainer cho mô hình: {type(model).__name__}...")

    # Chuyển X_test_sc về DataFrame để có tên cột khi vẽ SHAP
    X_test_df = pd.DataFrame(X_test_sc, columns=[f.replace('X_', '') for f in feature_names])

    # Khởi tạo Explainer (TreeExplainer phù hợp cho XGBoost/RF)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    # 1. SHAP Summary Plot (Beeswarm)
    print("[SHAP] 1. Đang tạo Summary Plot (Beeswarm)...")
    plt.figure(figsize=(12, 8), facecolor='#fcfcfc')
    shap.summary_plot(shap_values, X_test_df, show=False, plot_size=None, alpha=0.7)
    
    # Tinh chỉnh tiêu đề và lưới
    plt.title('SHAP Summary Plot: Feature Impact on Model Output', fontsize=16, fontweight='bold', pad=25, color='#2c3e50')
    plt.grid(True, linestyle='--', alpha=0.6, axis='x')
    plt.gca().set_facecolor('#fcfcfc')
    
    summary_path = os.path.join(shap_dir, 'shap_summary_plot.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      → Đã lưu: {summary_path}")

    # 2. SHAP Bar Plot (Global Importance)
    print("[SHAP] 2. Đang tạo Bar Plot (Global Importance)...")
    plt.figure(figsize=(12, 8), facecolor='#fcfcfc')
    # shap.summary_plot với plot_type="bar"
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False, color='#3498db', plot_size=None)
    
    plt.title('SHAP Global Feature Importance (Mean |SHAP value|)', fontsize=16, fontweight='bold', pad=25, color='#2c3e50')
    plt.xlabel('mean(|SHAP value|) (Ảnh hưởng trung bình)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4, axis='x')
    plt.gca().set_facecolor('#fcfcfc')
    
    bar_path = os.path.join(shap_dir, 'shap_bar_plot.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      → Đã lưu: {bar_path}")

    # 3. SHAP Waterfall Plot cho 1 quan sát đầu tiên (Local explanation)
    try:
        print("[SHAP] 3. Đang tạo Waterfall Plot cho mẫu đầu tiên...")
        # Lấy explanation object
        explainer_exp = shap.Explainer(model, X_test_df)
        shap_values_exp = explainer_exp(X_test_df)
        
        fig = plt.figure(figsize=(12, 7), facecolor='#fcfcfc')
        shap.plots.waterfall(shap_values_exp[0], show=False)
        
        plt.title('SHAP Waterfall Plot: Local Explanation for Instance #0', 
                  fontsize=16, fontweight='bold', pad=30, color='#2c3e50')
        
        waterfall_path = os.path.join(shap_dir, 'shap_waterfall_local.png')
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      → Đã lưu: {waterfall_path}")
    except Exception as e:
        print(f"      ⚠ Không thể tạo Waterfall Plot: {e}")

    print(f"\n[XONG] Tất cả kết quả SHAP đã được lưu tại: {shap_dir}")
