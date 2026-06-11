import matplotlib.pyplot as plt
import numpy as np
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

labels = ['Tập Huấn luyện (Train=857)', 'Tập Kiểm thử (Test=215)']
normal = np.array([489, 123])
anomaly = np.array([368, 92])

x = np.arange(len(labels))
width = 0.4

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, normal, width, label='Bình thường (Label=0)', color='#3498db', edgecolor='black')
rects2 = ax.bar(x + width/2, anomaly, width, label='Bất thường (Label=1)', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Số lượng quan sát', fontsize=11, fontweight='bold')
ax.set_title('PHÂN KIỔ QUAN SÁT VÀ TỶ LỆ NHÃN TRÊN TẬP HUẤN LUYỆN & KIỂM THỬ', pad=20, fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold', fontsize=11)
ax.legend(loc='upper right', fontsize=10)

ax.bar_label(rects1, padding=3, fontweight='bold')
ax.bar_label(rects2, padding=3, fontweight='bold')

# Chèn % vào giữa cột
for i, rect in enumerate(rects1):
    height = rect.get_height()
    total = normal[i] + anomaly[i]
    pct = height / total * 100
    ax.annotate(f"{pct:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height/2),
                xytext=(0, 0), textcoords="offset points",
                ha='center', va='center', color='white', fontweight='bold', fontsize=10)

for i, rect in enumerate(rects2):
    height = rect.get_height()
    total = normal[i] + anomaly[i]
    pct = height / total * 100
    ax.annotate(f"{pct:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height/2),
                xytext=(0, 0), textcoords="offset points",
                ha='center', va='center', color='white', fontweight='bold', fontsize=10)

ax.set_ylim(0, 600)
plt.tight_layout()

file_path = os.path.join(OUTPUT_DIR, 'train_test_distribution.png')
plt.savefig(file_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Xong! Luu tai: {file_path}")
