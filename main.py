# -*- coding: utf-8 -*-
"""
Pipeline Khai Phá Dữ Liệu: Phát hiện Bất thường BCTC Ngành Bán lẻ
Chuẩn CRISP-DM | Model: Không sử dụng SMOTE (sử dụng class_weight/scale_pos_weight)
"""
import sys
import os

from src.utils import setup_env, setup_logger, separator
from src.preprocessing import load_data, calculate_mscore_and_label
from src.eda import run_eda
from src.feature_eng import preprocess_features
from src.train import train_and_tune_models
from src.evaluate import evaluate_models
from src.shap_analysis import run_shap_analysis


# Cấu hình đường dẫn
OUTPUT_DIR = './output'
DATA_FILE = 'data/processed/BCTC_Retail_Industry_2020_2025.xlsx'

def main():
    # 0. Setup môi trường và Logger
    setup_env(OUTPUT_DIR)
    log_file = setup_logger(OUTPUT_DIR)
    
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr

    try:
        # 1 & 2: Nạp dữ liệu & Tạo nhãn proxy
        df = load_data(DATA_FILE)
        df_valid, imbalance_pct, SPW = calculate_mscore_and_label(df)
        
        # 3: Khám phá dữ liệu (EDA)
        run_eda(df_valid=df_valid, output_dir=OUTPUT_DIR)
        
        # 4 & 5: Feature Engineering & Tiền xử lý
        X_train_sc, X_test_sc, y_train, y_test, feature_names, df_valid = preprocess_features(
            df_valid, imbalance_pct, SPW)
        
        # Vẽ biểu đồ tương quan sau xử lý
        run_eda(df_valid=df_valid, feature_names=feature_names, output_dir=OUTPUT_DIR)

        # 6: Huấn luyện & Fine-tuning
        models = train_and_tune_models(X_train_sc, y_train, X_test_sc, y_test, SPW)

        # 7: Đánh giá mô hình
        evaluate_models(models, X_test_sc, y_test, feature_names, output_dir=OUTPUT_DIR)
        
        # 8: Phân tích SHAP (với XGBoost)
        if 'XGBoost (Fine-tuned)' in models:
            run_shap_analysis(models['XGBoost (Fine-tuned)'], X_test_sc, feature_names, output_dir=OUTPUT_DIR)
        
        # Hoàn tất pipeline
        separator("HOÀN THÀNH PIPELINE")
        output_files = [
            'eda_class_distribution.png', 'eda_mscore_hist.png',
            'eda_correlation.png', 'eda_feature_boxplots.png',
            'eval_confusion_matrix.png', 'eval_roc_curve.png', 
            'eval_feature_importance_xgb.png', 'terminal_log.txt',
        ]
        print(f"Các file đã xuất trong thư mục {OUTPUT_DIR}:")
        for f in output_files:
            exists = "✓" if os.path.exists(os.path.join(OUTPUT_DIR, f)) else "✗"
            print(f"  {exists}  {f}")
        print("\nPipeline hoàn tất thành công!")

    except Exception as e:
        print(f"\n[LỖI]: {e}")
        
    finally:
        # Khôi phục stdout
        sys.stdout = getattr(_orig_stdout, 'streams', [sys.stdout])[0] if hasattr(_orig_stdout, 'streams') else _orig_stdout
        sys.stderr = getattr(_orig_stderr, 'streams', [sys.stderr])[0] if hasattr(_orig_stderr, 'streams') else _orig_stderr
        log_file.close()
        log_file_path = os.path.join(OUTPUT_DIR, 'terminal_log.txt')
        print(f"\n[LOG] Terminal log đã lưu tại: {log_file_path}")

if __name__ == '__main__':
    main()
