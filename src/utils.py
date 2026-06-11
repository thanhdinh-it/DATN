# -*- coding: utf-8 -*-
import sys
import io
import os
import warnings
import matplotlib

def setup_env(output_dir='./output'):
    """Thiết lập môi trường: utf-8, warnings, matplotlib backend và thư mục output."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')
    
    os.makedirs(output_dir, exist_ok=True)


class Tee(io.TextIOBase):
    """Class hỗ trợ ghi log đồng thời ra terminal và file text."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try: 
                s.flush()
            except Exception: 
                pass


def setup_logger(output_dir='./output'):
    """Chuyển hướng sys.stdout và sys.stderr qua logger Tee."""
    log_file_path = os.path.join(output_dir, 'terminal_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    return log_file


def separator(title: str):
    """In dòng phân cách với tiêu đề."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def safe_divide(numerator, denominator, fill_nan: float = float('nan')):
    """Chia an toàn, tránh lỗi chia cho 0. Trả về NumPy arrays hoặc Pandas Series."""
    import numpy as np
    denom = denominator.copy().astype(float)
    denom[denom == 0] = np.nan
    result = numerator.astype(float) / denom
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(fill_nan)
