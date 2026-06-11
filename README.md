Dự án này phục vụ cho khóa luận tốt nghiệp ngành Phân tích Dữ liệu Kinh doanh. Codebase tập trung vào việc xây dựng một pipeline chuẩn CRISP-DM để phát hiện bất thường trên Báo cáo tài chính của các doanh nghiệp ngành bán lẻ bằng mô hình học máy (Machine Learning).

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-completed-success)
## Đặc điểm nổi bật
- **Proxy Labeling:** Sử dụng mô hình Beneish M-Score làm nhãn proxy đại diện cho các bất thường.
- **Không dùng phương pháp sinh mẫu nhân tạo (No SMOTE):** Tối ưu hóa hiện tượng mất cân bằng dữ liệu thông qua tham số nội tại của thuật toán (`class_weight`, `scale_pos_weight`).
- **Feature Engineering mạnh mẽ:** Trích xuất 12 tỷ số tài chính cốt lõi và xử lý ngoại lệ cẩn thận (winsorize 1%-99%).
- **Modeling & Fine-tuning:** Ứng dụng GridSearchCV và RandomizedSearchCV để tối ưu Random Forest, Logistic Regression và XGBoost.
- **Reporting:** Toàn bộ log và đồ thị (heatmap, histogram, roc_curve, feature importance) được tự động sinh và lưu vào folder `output/`.

## Cấu trúc Repository

```text
├── data/
│   └── processed/             # Dữ liệu BCTC dạng excel (BCTC_Retail_Industry_2020_2025.xlsx)
├── notebooks/                 # Thư mục chứa các Jupyter notebook (Draft/EDA)
├── output/                    # Kết quả phân tích, mô hình, báo cáo log và đồ họa
├── src/                       # Source code chính của Pipeline
│   ├── eda.py                 # Mã nguồn vẽ các biểu đồ phân phối và tương quan
│   ├── evaluate.py            # Hàm đánh giá ROC, confusion matrix, feature importance
│   ├── feature_eng.py         # Mã nguồn chia features phân bổ (winsorize, StandardScale)
│   ├── preprocessing.py       # Code tải dữ liệu và tính Beneish M-Score
│   ├── train.py               # Code chạy GridSearch & Train model
│   └── utils.py               # Các hàm tiện dụng dùng chung cho logger, helper
├── .gitignore                 # Các tệp/thư mục không đưa lên GitHub (data, output)
├── main.py                    # Script chạy chính toàn bộ quá trình CRISP-DM
├── README.md                  # File tài liệu dự án
└── requirements.txt           # Danh sách các thư viện Python phụ thuộc
```

##  Hướng dẫn cài đặt và chạy Dự án

**Bước 1:** Clone kho lưu trữ về máy:
```bash
git clone https://github.com/thanhdinh-it/Khoaluantotnghiepv2.git
cd Khoaluantotnghiepv2
```

**Bước 2:** (Tham khảo) Tạo môi trường ảo và kích hoạt:
```bash
python -m venv venv
# Đối với Windows:
venv\Scripts\activate
# Đối với Linux/Mac:
source venv/bin/activate
```

**Bước 3:** Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

**Bước 4:** Bỏ dữ liệu vào thư mục:
Hãy chắc chắn rằng đã cung cấp file dữ liệu `BCTC_Retail_Industry_2020_2025.xlsx` vào đường dẫn `data/processed/`. (Thư mục `data/` đã được cấu hình trong `.gitignore` để bảo mật).

**Bước 5:** Khởi chạy Pipeline
```bash
python main.py
```
Sau khi chạy, xem thành quả báo cáo tại terminal và các ảnh report ở thư mục `output/`.

## ✍🏻 Tác giả
- GitHub: [@thanhdinh-it](https://github.com/thanhdinh-it)
- Sinh viên thực hiện Khóa luận tốt nghiệp chuyên ngành Phân tích Dữ liệu Kinh doanh.
