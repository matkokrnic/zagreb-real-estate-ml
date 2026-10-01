#!/usr/bin/env python3
"""
Njuškalo Klasifikacija Stanova v3 — OOF predikcije
- Svaki stan dobiva predikciju od modela koji ga NIJE vidio
- Pouzdana klasifikacija svih 7174 stanova
- GREEN/YELLOW/RED
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# ============================================================================
# KONFIGURACIJA
# ============================================================================

INPUT_FILE = 'apartments_clean_v2.csv'
MODELS_FILE = 'models_tuned.pkl'
OUTPUT_FILE = 'apartments_classified_v2.csv'
OUTPUT_GREEN = 'apartments_GREEN_DEALS_v2.csv'
RANDOM_STATE = 42
CV_FOLDS = 5

GREEN_THRESHOLD = 0.15
RED_THRESHOLD = -0.15

NON_FEATURE_COLS = [
    'oglas_id', 'url', 'title', 'lokacija', 'district',
    'price_eur', 'log_price', 'price_per_m2',
    'datum_objave', 'owner_name', 'scraped_at'
]

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# ============================================================================
# UČITAVANJE
# ============================================================================

print_section("UČITAVANJE PODATAKA I MODELA")

df = pd.read_csv(INPUT_FILE, encoding='utf-8')
print(f"  ✓ Dataset: {len(df)} stanova, {len(df.columns)} kolumni")

with open(MODELS_FILE, 'rb') as f:
    models = pickle.load(f)

with open('best_params.json', 'r') as f:
    best_params = json.load(f)

feature_cols = models['feature_cols']
X = df[feature_cols].copy()
y = df['log_price'].copy()

print(f"  ✓ Features: {len(feature_cols)}")

# ============================================================================
# OOF PREDIKCIJE
# ============================================================================

print_section(f"OOF PREDIKCIJE ({CV_FOLDS}-fold, svih {len(X)} stanova)")

print(f"\n  Svaki stan dobiva predikciju od modela koji ga NIJE vidio pri treniranju.")
print(f"  Ovo osigurava pouzdanu i nepristranu klasifikaciju.\n")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"  Fold {fold+1}/{CV_FOLDS}...")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr = y.iloc[train_idx]

    m_xgb = xgb.XGBRegressor(**best_params['xgboost'], random_state=RANDOM_STATE, verbosity=0)
    m_xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = m_xgb.predict(X_val)

    m_lgb = lgb.LGBMRegressor(**best_params['lightgbm'], random_state=RANDOM_STATE, verbose=-1)
    m_lgb.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(-1)])
    oof_lgb[val_idx] = m_lgb.predict(X_val)

    m_cat = CatBoostRegressor(**best_params['catboost'], random_seed=RANDOM_STATE, verbose=0)
    m_cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = m_cat.predict(X_val)

log_price_pred = (oof_xgb + oof_lgb + oof_cat) / 3

r2_oof = r2_score(y, log_price_pred)
r2_xgb_oof = r2_score(y, oof_xgb)
r2_lgb_oof = r2_score(y, oof_lgb)
r2_cat_oof = r2_score(y, oof_cat)

print(f"\n  ✓ OOF R² po modelu:")
print(f"    XGBoost:  {r2_xgb_oof:.4f}")
print(f"    LightGBM: {r2_lgb_oof:.4f}")
print(f"    CatBoost: {r2_cat_oof:.4f}")
print(f"    Average:  {r2_oof:.4f}")

price_pred = np.exp(log_price_pred)
price_actual = df['price_eur'].values

# ============================================================================
# KLASIFIKACIJA
# ============================================================================

print_section("KLASIFIKACIJA GREEN / YELLOW / RED")

value_ratio = (price_pred - price_actual) / price_actual

def classify(ratio):
    if ratio >= GREEN_THRESHOLD:
        return 'GREEN'
    elif ratio <= RED_THRESHOLD:
        return 'RED'
    else:
        return 'YELLOW'

classification = [classify(r) for r in value_ratio]

# ============================================================================
# REZULTATI
# ============================================================================

print_section("SASTAVLJANJE REZULTATA")

results = df[['oglas_id', 'url', 'title', 'lokacija', 'district',
              'price_eur', 'm2', 'rooms', 'floor', 'year_built',
              'is_novogradnja', 'has_lift', 'has_parking', 'has_garage',
              'has_balcony', 'has_terrace', 'energetski_razred_num',
              'datum_objave']].copy()

results['predicted_price'] = np.round(price_pred, -2)
results['price_difference'] = np.round(price_pred - price_actual, -2)
results['value_ratio_pct'] = np.round(value_ratio * 100, 1)
results['classification'] = classification

energy_map_reverse = {7: 'A+', 6: 'A', 5: 'B', 4: 'C', 3: 'D', 2: 'E', 1: 'F', 0: 'G'}
results['energetski_razred'] = df['energetski_razred_num'].round().astype(int).map(energy_map_reverse)

results = results.sort_values('value_ratio_pct', ascending=False)

counts = results['classification'].value_counts()
print(f"\n  Distribucija klasifikacija:")
for cls in ['GREEN', 'YELLOW', 'RED']:
    count = counts.get(cls, 0)
    pct = count / len(results) * 100
    bar = '█' * int(pct / 2)
    print(f"    {cls:<8}: {count:5d} ({pct:5.1f}%)  {bar}")

# ============================================================================
# STATISTIKE PO KATEGORIJI
# ============================================================================

print_section("STATISTIKE PO KATEGORIJI")

for cls in ['GREEN', 'YELLOW', 'RED']:
    subset = results[results['classification'] == cls]
    if len(subset) == 0:
        continue
    print(f"\n  {cls} ({len(subset)} stanova):")
    print(f"    Prosječna cijena:      €{subset['price_eur'].mean():>10,.0f}")
    print(f"    Prosječna predviđena:  €{subset['predicted_price'].mean():>10,.0f}")
    print(f"    Prosječna razlika:     €{subset['price_difference'].mean():>+10,.0f}")
    print(f"    Prosječni value ratio:  {subset['value_ratio_pct'].mean():>+9.1f}%")
    print(f"    Prosječni m²:          {subset['m2'].mean():>10.0f} m²")

# ============================================================================
# TOP GREEN DEALOVI
# ============================================================================

print_section("TOP 20 GREEN DEALOVA")

green_deals = results[results['classification'] == 'GREEN'].head(20)
print(f"\n  {'#':<3} {'Lokacija':<35} {'Cijena':>10} {'Predviđeno':>12} {'Razlika':>10} {'%':>7}")
print(f"  {'-'*80}")

for i, (_, row) in enumerate(green_deals.iterrows(), 1):
    lokacija = str(row['lokacija'])[:33] if pd.notna(row['lokacija']) else 'N/A'
    print(f"  {i:<3} {lokacija:<35} €{row['price_eur']:>9,.0f} €{row['predicted_price']:>11,.0f} €{row['price_difference']:>+9,.0f} {row['value_ratio_pct']:>+6.1f}%")

# ============================================================================
# ANALIZA PO KVARTU
# ============================================================================

print_section("ANALIZA PO KVARTU")

district_analysis = results.groupby('district').agg(
    ukupno=('classification', 'count'),
    green=('classification', lambda x: (x == 'GREEN').sum()),
    yellow=('classification', lambda x: (x == 'YELLOW').sum()),
    red=('classification', lambda x: (x == 'RED').sum()),
    avg_ratio=('value_ratio_pct', 'mean'),
    avg_price=('price_eur', 'mean')
).sort_values('avg_ratio', ascending=False)

district_analysis['green_pct'] = (district_analysis['green'] / district_analysis['ukupno'] * 100).round(1)

print(f"\n  {'Kvart':<35} {'Ukupno':>8} {'GREEN':>7} {'%':>6} {'Avg Ratio':>10} {'Avg Cijena':>12}")
print(f"  {'-'*80}")
for district, row in district_analysis.iterrows():
    print(f"  {str(district):<35} {row['ukupno']:>8.0f} {row['green']:>7.0f} {row['green_pct']:>5.1f}% {row['avg_ratio']:>+9.1f}% €{row['avg_price']:>11,.0f}")

# ============================================================================
# SPREMANJE
# ============================================================================

print_section("SPREMANJE REZULTATA")

results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"  ✓ Svi stanovi: {OUTPUT_FILE} ({len(results)} redova)")

green_only = results[results['classification'] == 'GREEN'].copy()
green_only.to_csv(OUTPUT_GREEN, index=False, encoding='utf-8')
print(f"  ✓ Green dealovi: {OUTPUT_GREEN} ({len(green_only)} redova)")

summary = {
    'total_apartments': len(results),
    'green': int(counts.get('GREEN', 0)),
    'yellow': int(counts.get('YELLOW', 0)),
    'red': int(counts.get('RED', 0)),
    'green_pct': round(counts.get('GREEN', 0) / len(results) * 100, 1),
    'yellow_pct': round(counts.get('YELLOW', 0) / len(results) * 100, 1),
    'red_pct': round(counts.get('RED', 0) / len(results) * 100, 1),
    'oof_r2': round(r2_oof, 4),
    'green_threshold_pct': GREEN_THRESHOLD * 100,
    'red_threshold_pct': RED_THRESHOLD * 100
}

with open('classification_summary_v2.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ Summary: classification_summary_v2.json")

print(f"\n{'='*70}")
print(f"  KLASIFIKACIJA ZAVRŠENA!")
print(f"  GREEN: {counts.get('GREEN', 0)} | YELLOW: {counts.get('YELLOW', 0)} | RED: {counts.get('RED', 0)}")
print(f"  OOF R²: {r2_oof:.4f}")
print(f"{'='*70}\n")