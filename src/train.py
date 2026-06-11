# -*- coding: utf-8 -*-
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils import separator

def train_and_tune_models(X_train_sc, y_train, X_test_sc, y_test, SPW):
    """Huấn luyện GridSearchCV và RandomizedSearchCV."""
    separator("BƯỚC 6: HUẤN LUYỆN MÔ HÌNH VỚI GridSearchCV")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 6-A: Logistic Regression
    print("\n[6.1] Logistic Regression (class_weight='balanced')…")
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=2000, solver='lbfgs',
                           class_weight='balanced', random_state=42),
        param_grid={'C': [0.1, 1, 10]},
        cv=cv, scoring='f1', n_jobs=-1, verbose=0
    )
    lr_grid.fit(X_train_sc, y_train)
    print(f"      Best params: {lr_grid.best_params_} | CV F1: {lr_grid.best_score_:.4f}")

    # 6-B: Random Forest
    print("\n[6.2] Random Forest (class_weight='balanced')…")
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid={'max_depth': [5, 10, None], 'n_estimators': [100, 200]},
        cv=cv, scoring='f1', n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train_sc, y_train)
    print(f"      Best params: {rf_grid.best_params_} | CV F1: {rf_grid.best_score_:.4f}")

    # 6-C: XGBoost
    print(f"\n[6.3] XGBoost (scale_pos_weight={SPW})…")
    xgb_grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                      scale_pos_weight=SPW, random_state=42, verbosity=0),
        param_grid={'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 7]},
        cv=cv, scoring='f1', n_jobs=-1, verbose=0
    )
    xgb_grid.fit(X_train_sc, y_train)
    best_xgb_params = xgb_grid.best_params_
    print(f"      Best params: {best_xgb_params} | CV F1: {xgb_grid.best_score_:.4f}")

    
    separator("BƯỚC 6B: FINE-TUNING XGBoost (Best Model)")

    print(f"[6B] RandomizedSearchCV (n_iter=50) | Anchor: {best_xgb_params}")
    _lr = best_xgb_params['learning_rate']
    _md = best_xgb_params['max_depth']

    xgb_fine_search = RandomizedSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                      scale_pos_weight=SPW,
                      learning_rate=_lr, max_depth=_md,
                      random_state=42, verbosity=0),
        param_distributions={
            'n_estimators':     randint(100, 501),
            'min_child_weight': randint(1, 8),
            'subsample':        uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma':            uniform(0, 0.5),
            'reg_alpha':        uniform(0, 1.0),
            'reg_lambda':       uniform(0.5, 2.0),
        },
        n_iter=50, cv=cv, scoring='f1',
        random_state=42, n_jobs=-1, verbose=0
    )
    xgb_fine_search.fit(X_train_sc, y_train)
    print(f"     Fine-tuned Best params: {xgb_fine_search.best_params_}")
    print(f"     Fine-tuned CV F1: {xgb_fine_search.best_score_:.4f}  "
          f"(thô: {xgb_grid.best_score_:.4f}  "
          f"cải thiện: {(xgb_fine_search.best_score_ - xgb_grid.best_score_)*100:+.2f}%)")

    xgb_best_final = xgb_fine_search.best_estimator_

    # ── Bảng so sánh ─────────────────────────────
    print("\n" + "─" * 67)
    print("  SO SÁNH XGBoost: Trước vs Sau Fine-Tuning (tập Test)")
    print("─" * 67)
    _cols   = ['Mô hình', 'CV F1', 'Accuracy', 'Precision(1)', 'Recall(1)', 'F1(1)']
    _widths = [22, 8, 10, 13, 11, 8]
    print("  ".join(f"{c:<{w}}" for c, w in zip(_cols, _widths)))
    print("  ".join("─" * w for w in _widths))
    for label, mdl, cv_f1 in [("XGBoost (Trước)",    xgb_grid.best_estimator_, xgb_grid.best_score_),
                              ("XGBoost (Fine-tuned)", xgb_best_final,          xgb_fine_search.best_score_)]:
        yp = mdl.predict(X_test_sc)
        row = [label, f"{cv_f1:.4f}", f"{accuracy_score(y_test,yp):.4f}",
               f"{precision_score(y_test,yp):.4f}", f"{recall_score(y_test,yp):.4f}",
               f"{f1_score(y_test,yp):.4f}"]
        print("  ".join(f"{v:<{w}}" for v, w in zip(row, _widths)))
    print("─" * 67)

    models = {
        'Logistic Regression': lr_grid.best_estimator_,
        'Random Forest':       rf_grid.best_estimator_,
        'XGBoost (Fine-tuned)': xgb_best_final,
    }
    return models
