#!/usr/bin/env python3
"""
Njuškalo Model Training v2
- Ridge baseline
- CatBoost, XGBoost, LightGBM
- Stacking (CatBoost + XGBoost + LightGBM → Ridge meta-model)
- log(price) target
- Cross-validation
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# ============================================================================
# KONFIGURACIJA
# ============================================================================

INPUT_FILE = 'apartments_clean_v2.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Kolumne koje NE idu u model
NON_FEATURE_COLS = [
    'oglas_id', 'url', 'title', 'lokacija', 'district',
    'price_eur', 'log_price', 'price_per_m2',
    'datum_objave', 'owner_name', 'scraped_at'
]

# ============================================================================
# HELPER FUNKCIJE
# ============================================================================

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def evaluate_model(name, y_true, y_pred_log, prefix=""):
    """Evaluiraj model — konvertira log predikcije nazad u EUR"""
    y_pred_eur = np.expm1(y_pred_log) if y_pred_log.min() > 5 else np.exp(y_pred_log)
    y_true_eur = np.exp(y_true)

    r2 = r2_score(y_true, y_pred_log)
    mae = mean_absolute_error(y_true_eur, y_pred_eur)
    rmse = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))
    mape = np.mean(np.abs((y_true_eur - y_pred_eur) / y_true_eur)) * 100

    print(f"\n  {prefix}{name}:")
    print(f"    R²:   {r2:.4f}")
    print(f"    MAE:  €{mae:,.0f}")
    print(f"    RMSE: €{rmse:,.0f}")
    print(f"    MAPE: {mape:.1f}%")

    return r2, mae, rmse

# ============================================================================
# UČITAVANJE PODATAKA
# ============================================================================

print_section("UČITAVANJE PODATAKA")

df = pd.read_csv(INPUT_FILE, encoding='utf-8')
print(f"  ✓ Loaded: {len(df)} rows, {len(df.columns)} columns")

# Features i target
feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS]
X = df[feature_cols].copy()
y = df['log_price'].copy()

print(f"  ✓ Features: {len(feature_cols)}")
print(f"  ✓ Target: log_price (range {y.min():.2f} - {y.max():.2f})")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# RIDGE BASELINE
# ============================================================================

print_section("1. RIDGE BASELINE")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

r2_ridge, mae_ridge, rmse_ridge = evaluate_model("Ridge", y_test, y_pred_ridge)

# CV score
cv_scores_ridge = cross_val_score(
    Ridge(alpha=1.0),
    X_train_scaled, y_train,
    cv=CV_FOLDS, scoring='r2'
)
print(f"    CV R² ({CV_FOLDS}-fold): {cv_scores_ridge.mean():.4f} ± {cv_scores_ridge.std():.4f}")

# ============================================================================
# CATBOOST
# ============================================================================

print_section("2. CATBOOST")

catboost_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    random_seed=RANDOM_STATE,
    verbose=100,
    early_stopping_rounds=50
)

catboost_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test)
)

y_pred_catboost = catboost_model.predict(X_test)
r2_catboost, mae_catboost, rmse_catboost = evaluate_model("CatBoost", y_test, y_pred_catboost)

# ============================================================================
# XGBOOST
# ============================================================================

print_section("3. XGBOOST")

xgboost_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    early_stopping_rounds=50,
    verbosity=1
)

xgboost_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

y_pred_xgboost = xgboost_model.predict(X_test)
r2_xgboost, mae_xgboost, rmse_xgboost = evaluate_model("XGBoost", y_test, y_pred_xgboost)

# ============================================================================
# LIGHTGBM
# ============================================================================

print_section("4. LIGHTGBM")

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'verbose': -1
}

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_eval],
    callbacks=callbacks
)

y_pred_lgb = lgb_model.predict(X_test)
r2_lgb, mae_lgb, rmse_lgb = evaluate_model("LightGBM", y_test, y_pred_lgb)

# ============================================================================
# STACKING
# ============================================================================

print_section("5. STACKING (CatBoost + XGBoost + LightGBM → Ridge)")

print("\n  Generating out-of-fold predictions...")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Out-of-fold predikcije za trening meta-modela
oof_catboost = np.zeros(len(X_train))
oof_xgboost = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))

# Test predikcije (average po foldovima)
test_catboost = np.zeros(len(X_test))
test_xgboost = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"  Fold {fold+1}/{CV_FOLDS}...")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # CatBoost fold
    cb_fold = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        random_seed=RANDOM_STATE, verbose=0
    )
    cb_fold.fit(X_tr, y_tr)
    oof_catboost[val_idx] = cb_fold.predict(X_val)
    test_catboost += cb_fold.predict(X_test) / CV_FOLDS

    # XGBoost fold
    xgb_fold = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        random_state=RANDOM_STATE, verbosity=0
    )
    xgb_fold.fit(X_tr, y_tr)
    oof_xgboost[val_idx] = xgb_fold.predict(X_val)
    test_xgboost += xgb_fold.predict(X_test) / CV_FOLDS

    # LightGBM fold
    lgb_tr = lgb.Dataset(X_tr, label=y_tr)
    lgb_val_ds = lgb.Dataset(X_val, label=y_val, reference=lgb_tr)
    lgb_fold = lgb.train(
        {**lgb_params, 'verbose': -1},
        lgb_tr,
        num_boost_round=500,
        valid_sets=[lgb_val_ds],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)]
    )
    oof_lgb[val_idx] = lgb_fold.predict(X_val)
    test_lgb += lgb_fold.predict(X_test) / CV_FOLDS

# Meta-features
X_meta_train = np.column_stack([oof_catboost, oof_xgboost, oof_lgb])
X_meta_test = np.column_stack([test_catboost, test_xgboost, test_lgb])

# Ridge meta-model
meta_scaler = StandardScaler()
X_meta_train_scaled = meta_scaler.fit_transform(X_meta_train)
X_meta_test_scaled = meta_scaler.transform(X_meta_test)

meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta_train_scaled, y_train)

y_pred_stack = meta_model.predict(X_meta_test_scaled)
r2_stack, mae_stack, rmse_stack = evaluate_model("Stacking", y_test, y_pred_stack)

# CV score na meta razini
cv_meta = cross_val_score(
    Ridge(alpha=1.0),
    X_meta_train_scaled, y_train,
    cv=CV_FOLDS, scoring='r2'
)
print(f"    Meta-model CV R²: {cv_meta.mean():.4f} ± {cv_meta.std():.4f}")

# ============================================================================
# USPOREDBA MODELA
# ============================================================================

print_section("USPOREDBA SVIH MODELA")

results = {
    'Ridge (baseline)': r2_ridge,
    'CatBoost': r2_catboost,
    'XGBoost': r2_xgboost,
    'LightGBM': r2_lgb,
    'Stacking': r2_stack
}

print(f"\n  {'Model':<25} {'R²':>10} {'vs Ridge':>10}")
print(f"  {'-'*45}")
for name, r2 in sorted(results.items(), key=lambda x: x[1], reverse=True):
    improvement = f"+{(r2 - r2_ridge)*100:.1f}%" if name != 'Ridge (baseline)' else "baseline"
    marker = " ← BEST" if r2 == max(results.values()) else ""
    print(f"  {name:<25} {r2:>10.4f} {improvement:>10}{marker}")

# ============================================================================
# FEATURE IMPORTANCE (CatBoost)
# ============================================================================

print_section("TOP 20 FEATURE IMPORTANCE (CatBoost)")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': catboost_model.get_feature_importance()
}).sort_values('importance', ascending=False)

print()
for _, row in feature_importance.head(20).iterrows():
    bar = '█' * int(row['importance'] / feature_importance['importance'].max() * 30)
    print(f"  {row['feature']:<40} {row['importance']:>8.2f}  {bar}")

# ============================================================================
# SPREMANJE MODELA
# ============================================================================

print_section("SPREMANJE MODELA")

best_model_name = max(results, key=results.get)
print(f"\n  Best model: {best_model_name} (R² = {results[best_model_name]:.4f})")

# Spremi sve modele
models_to_save = {
    'ridge': {'model': ridge, 'scaler': scaler},
    'catboost': catboost_model,
    'xgboost': xgboost_model,
    'lightgbm': lgb_model,
    'stacking': {
        'meta_model': meta_model,
        'meta_scaler': meta_scaler,
        'feature_cols': feature_cols
    },
    'results': results,
    'feature_importance': feature_importance.to_dict()
}

with open('models_v2.pkl', 'wb') as f:
    pickle.dump(models_to_save, f)
print(f"  ✓ Saved: models_v2.pkl")

# Spremi feature importance
feature_importance.to_csv('feature_importance_v2.csv', index=False)
print(f"  ✓ Saved: feature_importance_v2.csv")

# Spremi rezultate
results_df = pd.DataFrame([
    {'model': name, 'r2': r2, 'mae': mae, 'rmse': rmse}
    for name, (r2, mae, rmse) in [
        ('Ridge', (r2_ridge, mae_ridge, rmse_ridge)),
        ('CatBoost', (r2_catboost, mae_catboost, rmse_catboost)),
        ('XGBoost', (r2_xgboost, mae_xgboost, rmse_xgboost)),
        ('LightGBM', (r2_lgb, mae_lgb, rmse_lgb)),
        ('Stacking', (r2_stack, mae_stack, rmse_stack)),
    ]
])
results_df.to_csv('model_results_v2.csv', index=False)
print(f"  ✓ Saved: model_results_v2.csv")

print(f"\n{'='*70}")
print(f"  TRENIRANJE ZAVRŠENO!")
print(f"  Best R²: {results[best_model_name]:.4f} ({best_model_name})")
print(f"{'='*70}\n")