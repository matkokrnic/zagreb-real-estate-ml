#!/usr/bin/env python3
"""
Hyperparameter tuning s Optuna
- XGBoost tuning
- LightGBM tuning
- Treniranje finalnih modela s najboljim parametrima
- Usporedba s prethodnim rezultatima
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
N_TRIALS = 100  # Broj Optuna pokušaja po modelu

NON_FEATURE_COLS = [
    'oglas_id', 'url', 'title', 'lokacija', 'district',
    'price_eur', 'log_price', 'price_per_m2',
    'datum_objave', 'owner_name', 'scraped_at'
]

# Prethodni rezultati za usporedbu
PREVIOUS_RESULTS = {
    'Ridge (baseline)': 0.8274,
    'CatBoost': 0.8910,
    'XGBoost': 0.8978,
    'LightGBM': 0.8963,
    'Stacking': 0.8947
}

# ============================================================================
# HELPER
# ============================================================================

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def evaluate(name, y_true, y_pred_log):
    y_pred_eur = np.exp(y_pred_log)
    y_true_eur = np.exp(y_true)
    r2 = r2_score(y_true, y_pred_log)
    mae = mean_absolute_error(y_true_eur, y_pred_eur)
    rmse = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))
    mape = np.mean(np.abs((y_true_eur - y_pred_eur) / y_true_eur)) * 100
    print(f"\n  {name}:")
    print(f"    R²:   {r2:.4f}")
    print(f"    MAE:  €{mae:,.0f}")
    print(f"    RMSE: €{rmse:,.0f}")
    print(f"    MAPE: {mape:.1f}%")
    return r2, mae, rmse

# ============================================================================
# UČITAVANJE
# ============================================================================

print_section("UČITAVANJE PODATAKA")

df = pd.read_csv(INPUT_FILE, encoding='utf-8')
feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS]
X = df[feature_cols].copy()
y = df['log_price'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"  ✓ Features: {len(feature_cols)}")
print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# XGBOOST TUNING
# ============================================================================

print_section(f"XGBOOST TUNING ({N_TRIALS} trials)")

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': RANDOM_STATE,
        'verbosity': 0
    }

    # Cross-validation score
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  verbose=False)
        scores.append(r2_score(y_val, model.predict(X_val)))

    return np.mean(scores)

print(f"\n  Pokretanje {N_TRIALS} trials za XGBoost...")
print(f"  (ovo može trajati 15-25 minuta)\n")

xgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n  ✓ Best XGBoost CV R²: {xgb_study.best_value:.4f}")
print(f"  ✓ Best params:")
for k, v in xgb_study.best_params.items():
    print(f"    {k}: {v}")

# Treniraj finalni XGBoost s najboljim parametrima
best_xgb_params = {**xgb_study.best_params, 'random_state': RANDOM_STATE, 'verbosity': 0}
xgb_tuned = xgb.XGBRegressor(**best_xgb_params)
xgb_tuned.fit(X_train, y_train)
y_pred_xgb_tuned = xgb_tuned.predict(X_test)
r2_xgb_tuned, mae_xgb_tuned, rmse_xgb_tuned = evaluate("XGBoost Tuned", y_test, y_pred_xgb_tuned)

# ============================================================================
# LIGHTGBM TUNING
# ============================================================================

print_section(f"LIGHTGBM TUNING ({N_TRIALS} trials)")

def lgb_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': RANDOM_STATE,
        'verbose': -1
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)])
        scores.append(r2_score(y_val, model.predict(X_val)))

    return np.mean(scores)

print(f"\n  Pokretanje {N_TRIALS} trials za LightGBM...")
print(f"  (ovo može trajati 15-25 minuta)\n")

lgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n  ✓ Best LightGBM CV R²: {lgb_study.best_value:.4f}")
print(f"  ✓ Best params:")
for k, v in lgb_study.best_params.items():
    print(f"    {k}: {v}")

# Treniraj finalni LightGBM
best_lgb_params = {**lgb_study.best_params, 'random_state': RANDOM_STATE, 'verbose': -1}
lgb_tuned = lgb.LGBMRegressor(**best_lgb_params)
lgb_tuned.fit(X_train, y_train)
y_pred_lgb_tuned = lgb_tuned.predict(X_test)
r2_lgb_tuned, mae_lgb_tuned, rmse_lgb_tuned = evaluate("LightGBM Tuned", y_test, y_pred_lgb_tuned)

# ============================================================================
# CATBOOST TUNING
# ============================================================================

print_section(f"CATBOOST TUNING ({N_TRIALS} trials)")

def catboost_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'random_seed': RANDOM_STATE,
        'verbose': 0
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        scores.append(r2_score(y_val, model.predict(X_val)))

    return np.mean(scores)

print(f"\n  Pokretanje {N_TRIALS} trials za CatBoost...")
print(f"  (ovo može trajati 20-30 minuta)\n")

cat_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
cat_study.optimize(catboost_objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n  ✓ Best CatBoost CV R²: {cat_study.best_value:.4f}")
print(f"  ✓ Best params:")
for k, v in cat_study.best_params.items():
    print(f"    {k}: {v}")

best_cat_params = {**cat_study.best_params, 'random_seed': RANDOM_STATE, 'verbose': 0}
cat_tuned = CatBoostRegressor(**best_cat_params)
cat_tuned.fit(X_train, y_train)
y_pred_cat_tuned = cat_tuned.predict(X_test)
r2_cat_tuned, mae_cat_tuned, rmse_cat_tuned = evaluate("CatBoost Tuned", y_test, y_pred_cat_tuned)

# ============================================================================
# STACKING S TUNED MODELIMA
# ============================================================================

print_section("STACKING S TUNED MODELIMA")

print("\n  Generating out-of-fold predictions...")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))

test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"  Fold {fold+1}/{CV_FOLDS}...")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # XGBoost
    m_xgb = xgb.XGBRegressor(**{**xgb_study.best_params, 'random_state': RANDOM_STATE, 'verbosity': 0})
    m_xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = m_xgb.predict(X_val)
    test_xgb += m_xgb.predict(X_test) / CV_FOLDS

    # LightGBM
    m_lgb = lgb.LGBMRegressor(**{**lgb_study.best_params, 'random_state': RANDOM_STATE, 'verbose': -1})
    m_lgb.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(-1)])
    oof_lgb[val_idx] = m_lgb.predict(X_val)
    test_lgb += m_lgb.predict(X_test) / CV_FOLDS

    # CatBoost
    m_cat = CatBoostRegressor(**{**cat_study.best_params, 'random_seed': RANDOM_STATE, 'verbose': 0})
    m_cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = m_cat.predict(X_val)
    test_cat += m_cat.predict(X_test) / CV_FOLDS

# Meta-model
X_meta_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
X_meta_test = np.column_stack([test_xgb, test_lgb, test_cat])

meta_scaler = StandardScaler()
X_meta_train_s = meta_scaler.fit_transform(X_meta_train)
X_meta_test_s = meta_scaler.transform(X_meta_test)

meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta_train_s, y_train)
y_pred_stack_tuned = meta_model.predict(X_meta_test_s)
r2_stack_tuned, mae_stack_tuned, rmse_stack_tuned = evaluate("Stacking Tuned", y_test, y_pred_stack_tuned)

# ============================================================================
# FINALNA USPOREDBA
# ============================================================================

print_section("FINALNA USPOREDBA — PRIJE vs POSLIJE TUNINGA")

new_results = {
    'XGBoost Tuned': r2_xgb_tuned,
    'LightGBM Tuned': r2_lgb_tuned,
    'CatBoost Tuned': r2_cat_tuned,
    'Stacking Tuned': r2_stack_tuned,
}

print(f"\n  {'Model':<25} {'R² Prije':>10} {'R² Poslije':>12} {'Poboljšanje':>12}")
print(f"  {'-'*60}")

for name, r2_new in sorted(new_results.items(), key=lambda x: x[1], reverse=True):
    base_name = name.replace(' Tuned', '')
    r2_old = PREVIOUS_RESULTS.get(base_name, None)
    if r2_old:
        improvement = f"+{(r2_new - r2_old)*100:.2f}%"
    else:
        improvement = "N/A"
    marker = " ← BEST" if r2_new == max(new_results.values()) else ""
    print(f"  {name:<25} {r2_old if r2_old else 'N/A':>10} {r2_new:>12.4f} {improvement:>12}{marker}")

print(f"\n  Prethodni best: XGBoost         R² = {PREVIOUS_RESULTS['XGBoost']:.4f}")
print(f"  Novi best:      {max(new_results, key=new_results.get):<15} R² = {max(new_results.values()):.4f}")
print(f"  Ukupno poboljšanje: +{(max(new_results.values()) - PREVIOUS_RESULTS['XGBoost'])*100:.2f}%")

# ============================================================================
# SPREMANJE
# ============================================================================

print_section("SPREMANJE")

best_model_name = max(new_results, key=new_results.get)
print(f"\n  Best model: {best_model_name} (R² = {new_results[best_model_name]:.4f})")

tuned_models = {
    'xgboost_tuned': xgb_tuned,
    'lightgbm_tuned': lgb_tuned,
    'catboost_tuned': cat_tuned,
    'stacking_tuned': {
        'meta_model': meta_model,
        'meta_scaler': meta_scaler,
    },
    'best_params': {
        'xgboost': xgb_study.best_params,
        'lightgbm': lgb_study.best_params,
        'catboost': cat_study.best_params,
    },
    'results': new_results,
    'feature_cols': feature_cols
}

with open('models_tuned.pkl', 'wb') as f:
    pickle.dump(tuned_models, f)
print(f"  ✓ Saved: models_tuned.pkl")

# Spremi best params kao JSON
import json
with open('best_params.json', 'w') as f:
    json.dump({
        'xgboost': xgb_study.best_params,
        'lightgbm': lgb_study.best_params,
        'catboost': cat_study.best_params,
    }, f, indent=2)
print(f"  ✓ Saved: best_params.json")

print(f"\n{'='*70}")
print(f"  TUNING ZAVRŠEN!")
print(f"  Best R²: {max(new_results.values()):.4f} ({max(new_results, key=new_results.get)})")
print(f"{'='*70}\n")