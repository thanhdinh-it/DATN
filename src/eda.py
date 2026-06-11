# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.utils import separator

def run_eda(df_valid, feature_names=None, output_dir='./output'):
    """Vẽ và lưu kết quả Phân tích EDA cho Dataset với thẩm mỹ cao."""
    separator("BƯỚC 3: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)")
    
    # Thiết lập phong cách chung
    sns.set_theme(style="whitegrid", context="notebook", palette="tab10")
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 3-A Class distribution
    print("[3.1] Lưu eda_class_distribution.png…")
    counts = df_valid['Anomaly_Label'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#fcfcfc')
    
    colors = ['#27ae60', '#e74c3c'] # Green for normal, Red for anomaly
    bars = ax.bar(['Bình thường (0)', 'Bất thường (1)'], counts.values,
                  color=colors, edgecolor='#2c3e50', linewidth=1.5, width=0.6)
    
    # Thêm nhãn số lượng trên đầu bar, tránh trùng chữ
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(counts.values)*0.02),
                f'{val:,}\n({val/len(df_valid)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2c3e50')
                
    ax.set_title('Phân phối Nhãn Bất thường (Anomaly_Label)', fontsize=16, fontweight='bold', pad=25)
    ax.set_ylabel('Số quan sát', fontsize=12, fontweight='bold')
    ax.set_xlabel('Phân loại', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(counts.values) * 1.2) # Tạo không gian cho text
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_class_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("      → Đã lưu.")

    # 3-B M-Score histogram
    print("[3.2] Lưu eda_mscore_hist.png…")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#fcfcfc')
    
    # Vẽ histogram chồng lên nhau với KDE
    sns.histplot(data=df_valid, x='M_Score', hue='Anomaly_Label', bins=50, 
                 palette={0: '#27ae60', 1: '#e74c3c'}, element="step", alpha=0.4, ax=ax, kde=True)
    
    ax.axvline(-2.22, color='#2c3e50', linestyle='--', linewidth=2.5, label='Ngưỡng Beneish (-2.22)')
    
    ax.set_title('Phân phối M-Score theo Nhóm', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Beneish M-Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tần số quan sát', fontsize=12, fontweight='bold')
    ax.set_xlim(-10, 5) # Tập trung vào dải dữ liệu chính
    
    # Custom legend để không trùng chữ
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(title='Trạng thái', labels=['Bình thường', 'Bất thường', 'Ngưỡng -2.22'], fontsize=10, loc='upper right')
    
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_mscore_hist.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("      → Đã lưu.")

    # Nếu có mảng feature_names, vẽ correlation heatmap
    if feature_names:
        print("[4.2] Lưu eda_correlation.png…")
        fig, ax = plt.subplots(figsize=(14, 11), facecolor='#ffffff')
        
        # Lấy các đặc trưng và làm sạch tên để hiển thị
        display_names = [c.replace('X_','') for c in feature_names]
        corr = df_valid[feature_names].corr()
        corr.columns = display_names
        corr.index = display_names
        
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        
        # Heatmap chuyên nghiệp
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap, center=0,
                    linewidths=1, ax=ax, square=True,
                    cbar_kws={"shrink": .8},
                    annot_kws={'size': 9, 'weight': 'bold'})
                    
        ax.set_title('Ma trận tương quan 12 Tỷ số tài chính (Pearson Correlation)', 
                     fontsize=18, fontweight='bold', pad=30)
        
        # Xoay nhãn để tránh trùng
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eda_correlation.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("      → Đã lưu.")

        # Thêm: 4-C Boxplot so sánh 2 dải nhãn
        print("[4.3] Lưu eda_feature_boxplots.png…")
        # Chia layout 3x4
        fig, axes = plt.subplots(3, 4, figsize=(18, 14), facecolor='#fcfcfc')
        axes = axes.flatten()
        
        df_plot = df_valid.copy()
        df_plot['Label_Name'] = df_plot['Anomaly_Label'].map({0: 'Bình thường', 1: 'Bất thường'})
        
        for i, col in enumerate(feature_names):
            sns.boxplot(x='Label_Name', y=col, data=df_plot, ax=axes[i], 
                        palette={'Bình thường': '#27ae60', 'Bất thường': '#e74c3c'},
                        showfliers=False, width=0.6, linewidth=1.5)
            
            axes[i].set_title(col.replace('X_',''), fontsize=13, fontweight='bold', color='#34495e')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].tick_params(labelsize=10)
            axes[i].grid(True, linestyle=':', alpha=0.6)
            
        fig.suptitle('Phân tích Đặc trưng theo Nhóm Nhãn (Boxplot - Đã ẩn Outliers)', 
                     fontsize=20, fontweight='bold', y=1.02, color='#2c3e50')
        
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(output_dir, 'eda_feature_boxplots.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("      → Đã lưu.")
