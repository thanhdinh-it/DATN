# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from src.utils import separator

def evaluate_models(models, X_test_sc, y_test, feature_names, output_dir='./output'):
    """Đánh giá mô hình trên tập test với đồ thị thẩm mỹ, chuyên nghiệp."""
    separator("BƯỚC 7: ĐÁNH GIÁ MÔ HÌNH")
    
    # Thiết lập phong cách chung
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    
    LABEL_NAMES = ['Bình thường (0)', 'Bất thường (1)']

    # 7-A) Classification reports
    print()
    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        y_pred = model.predict(X_test_sc)
        print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, digits=4))

    # 7-B) Confusion matrices
    print("[7.1] Lưu eval_confusion_matrix.png…")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#fcfcfc')
    
    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test_sc))
        # Sử dụng bảng màu Blues sang trọng
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['0','1'], yticklabels=['0','1'],
                    linewidths=2, linecolor='white', square=True,
                    cbar=False,
                    annot_kws={'size':16, 'weight':'bold'})
        
        ax.set_title(name, fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
        ax.set_xlabel('Dự đoán', fontsize=11, fontweight='bold')
        ax.set_ylabel('Thực tế', fontsize=11, fontweight='bold')
        
    plt.suptitle('Ma trận Nhầm lẫn (Confusion Matrix) - So sánh các mô hình', 
                 fontsize=18, fontweight='bold', y=1.05, color='#2c3e50')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_confusion_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("      → Đã lưu.")

    # 7-C) ROC curves
    print("[7.2] Lưu eval_roc_curve.png…")
    fig, ax = plt.subplots(figsize=(9, 7), facecolor='#fcfcfc')
    
    # Bảng màu cho 3 đường ROC
    colors = ['#1f77b4', '#2ca02c', '#d62728'] 
    
    for (name, model), color in zip(models.items(), colors):
        y_score = (model.predict_proba(X_test_sc)[:,1]
                   if hasattr(model,'predict_proba')
                   else model.decision_function(X_test_sc))
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=3, color=color, label=f'{name} (AUC = {roc_auc:.4f})')
        
    ax.plot([0,1],[0,1],'k--', lw=1.5, alpha=0.6, label='Ngẫu nhiên (Baseline)')
    
    ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.05])
    ax.set_xlabel('Tỷ lệ Dương tính giả (False Positive Rate)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tỷ lệ Dương tính thật (True Positive Rate)', fontsize=12, fontweight='bold')
    ax.set_title('Đường cong ROC (Receiver Operating Characteristic)', fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines[['top','right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_roc_curve.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("      → Đã lưu.")

    # 7-D) XGBoost Feature Importance
    if 'XGBoost (Fine-tuned)' in models:
        xgb_best_final = models['XGBoost (Fine-tuned)']
        print("[7.3] Lưu eval_feature_importance_xgb.png…")
        
        importances = xgb_best_final.feature_importances_
        feat_labels = [f.replace('X_','') for f in feature_names]
        sorted_idx  = np.argsort(importances)
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#fcfcfc')
        
        # Tạo dải màu từ xanh đến đỏ
        norm = plt.Normalize(importances.min(), importances.max())
        colors_bar = plt.cm.viridis(norm(importances[sorted_idx]))
        
        bars = ax.barh(range(len(sorted_idx)), importances[sorted_idx],
                color=colors_bar, edgecolor='#2c3e50', linewidth=1.0)
        
        # Thêm giá trị quan trọng lên thanh
        for i, v in enumerate(importances[sorted_idx]):
            ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feat_labels[i] for i in sorted_idx], fontsize=11, fontweight='bold')
        ax.set_xlabel('Mức độ quan trọng (Gain Score)', fontsize=12, fontweight='bold')
        ax.set_title('XGBoost (Fine-tuned) – Tầm quan trọng của các Đặc trưng', 
                     fontsize=16, fontweight='bold', pad=25)
        
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eval_feature_importance_xgb.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("      → Đã lưu.")
