import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from src.preprocessing import load_data, calculate_mscore_and_label
from src.feature_eng import preprocess_features

df = load_data('data/processed/BCTC_Retail_Industry_2020_2025.xlsx')
df_valid, _, SPW = calculate_mscore_and_label(df)
_, _, _, _, F, d = preprocess_features(df_valid, 0, SPW)

corr = d[F].corr()
print('Correlations > 0.75:')
drop_candidates = set()
for i in range(len(F)):
    for j in range(i+1, len(F)):
        if abs(corr.iloc[i,j]) > 0.75:
            print(f'{F[i]} - {F[j]}: {corr.iloc[i,j]:.3f}')
            drop_candidates.add(F[j])
print("Candidates to drop:", drop_candidates)
