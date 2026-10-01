#!/usr/bin/env python3
"""
Vizualizacije za završni rad
1. Predicted vs Actual scatter plot
2. Feature importance bar chart
3. Box plot cijena po kvartovima
4. Model progression bar chart
5. Korelacijska matrica ključnih značajki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STILOVI
# ============================================================================

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'

COLORS = {
    'primary': '#2563EB',
    'green': '#16A34A',
    'yellow': '#D97706',
    'red': '#DC2626',
    'gray': '#6B7280',
    'light': '#F3F4F6'
}

# ============================================================================
# UČITAVANJE
# ============================================================================

print("Učitavam podatke...")
df = pd.read_csv('apartments_clean_v2.csv', encoding='utf-8')
classified = pd.read_csv('apartments_classified_v2.csv', encoding='utf-8')
feature_imp = pd.read_csv('feature_importance_v2.csv', encoding='utf-8')

print(f"  ✓ Dataset: {len(df)} stanova")
print(f"  ✓ Classified: {len(classified)} stanova")

# ============================================================================
# 1. PREDICTED VS ACTUAL
# ============================================================================

print("\n1. Predicted vs Actual scatter plot...")

fig, ax = plt.subplots(1, 1, figsize=(9, 7))

# Boje po klasifikaciji
colors = classified['classification'].map({
    'GREEN': COLORS['green'],
    'YELLOW': COLORS['gray'],
    'RED': COLORS['red']
})

scatter = ax.scatter(
    classified['price_eur'] / 1000,
    classified['predicted_price'] / 1000,
    c=colors,
    alpha=0.4,
    s=15,
    linewidths=0
)

# Dijagonala (savršena predikcija)
max_val = max(classified['price_eur'].max(), classified['predicted_price'].max()) / 1000
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.6, label='Savršena predikcija')

# ±15% zone
x_range = np.linspace(0, max_val, 100)
ax.fill_between(x_range, x_range * 0.85, x_range * 1.15,
                alpha=0.08, color=COLORS['primary'], label='±15% zona')

ax.set_xlabel('Stvarna cijena (tisuće €)', fontsize=12)
ax.set_ylabel('Predviđena cijena (tisuće €)', fontsize=12)
ax.set_title('Predviđena vs. Stvarna cijena stanova u Zagrebu\n(Stacking model, R² = 0.88)', fontsize=13)

# Legenda
green_patch = mpatches.Patch(color=COLORS['green'], alpha=0.7, label=f"GREEN ({(classified['classification']=='GREEN').sum()})")
yellow_patch = mpatches.Patch(color=COLORS['gray'], alpha=0.7, label=f"YELLOW ({(classified['classification']=='YELLOW').sum()})")
red_patch = mpatches.Patch(color=COLORS['red'], alpha=0.7, label=f"RED ({(classified['classification']=='RED').sum()})")
diag_line = plt.Line2D([0], [0], color='k', linestyle='--', linewidth=1.5, label='Savršena predikcija')

ax.legend(handles=[green_patch, yellow_patch, red_patch, diag_line],
          loc='upper left', framealpha=0.9, fontsize=9)

# R² annotation
from sklearn.metrics import r2_score
r2 = r2_score(np.log(classified['price_eur']), np.log(classified['predicted_price']))
ax.text(0.97, 0.05, f'OOF R² = {r2:.3f}',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.8))

plt.tight_layout()
plt.savefig('viz_1_predicted_vs_actual.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Spremljeno: viz_1_predicted_vs_actual.png")

# ============================================================================
# 2. FEATURE IMPORTANCE
# ============================================================================

print("\n2. Feature importance bar chart...")

# Top 20 značajki
top_n = 20
top_features = feature_imp.nlargest(top_n, 'importance').sort_values('importance')

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

bars = ax.barh(
    range(len(top_features)),
    top_features['importance'],
    color=COLORS['primary'],
    alpha=0.8,
    edgecolor='white',
    linewidth=0.5
)

# Bojaj top 5 drugačije
for i, bar in enumerate(bars):
    if i >= len(top_features) - 5:
        bar.set_color('#1E40AF')
        bar.set_alpha(1.0)

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=9)
ax.set_xlabel('Važnost značajke (%)', fontsize=11)
ax.set_title(f'Top {top_n} najvažnijih značajki\n(CatBoost Feature Importance)', fontsize=13)

# Vrijednosti na barovima
for i, (val, bar) in enumerate(zip(top_features['importance'], bars)):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=8, color='#374151')

ax.set_xlim(0, top_features['importance'].max() * 1.15)

plt.tight_layout()
plt.savefig('viz_2_feature_importance.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Spremljeno: viz_2_feature_importance.png")

# ============================================================================
# 3. BOX PLOT CIJENA PO KVARTOVIMA
# ============================================================================

print("\n3. Box plot cijena po kvartovima...")

# Samo kvartovi s dovoljno podataka
district_counts = df['district'].value_counts()
valid_districts = district_counts[district_counts >= 30].index.tolist()
df_filtered = df[df['district'].isin(valid_districts)].copy()

# Sortiraj po medianu
district_medians = df_filtered.groupby('district')['price_eur'].median().sort_values(ascending=True)
district_order = district_medians.index.tolist()

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

data_by_district = [df_filtered[df_filtered['district'] == d]['price_eur'].values / 1000
                    for d in district_order]

bp = ax.boxplot(
    data_by_district,
    vert=False,
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2),
    flierprops=dict(marker='o', markersize=2, alpha=0.3, color=COLORS['gray']),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2)
)

# Gradijent boja od jeftinijih do skupljim kvartovima
n = len(district_order)
for i, patch in enumerate(bp['boxes']):
    alpha = 0.5 + 0.5 * (i / n)
    patch.set_facecolor(COLORS['primary'])
    patch.set_alpha(alpha)

ax.set_yticks(range(1, len(district_order) + 1))
ax.set_yticklabels(district_order, fontsize=9)
ax.set_xlabel('Cijena (tisuće €)', fontsize=11)
ax.set_title('Distribucija cijena stanova po gradskim četvrtima Zagreba\n(medijan, Q1-Q3, min-max)', fontsize=13)

# Dodaj broj oglasa
for i, district in enumerate(district_order):
    count = len(df_filtered[df_filtered['district'] == district])
    ax.text(ax.get_xlim()[1] * 0.98, i + 1, f'n={count}',
            va='center', ha='right', fontsize=7, color=COLORS['gray'])

plt.tight_layout()
plt.savefig('viz_3_boxplot_kvartovi.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Spremljeno: viz_3_boxplot_kvartovi.png")

# ============================================================================
# 4. MODEL PROGRESSION
# ============================================================================

print("\n4. Model progression bar chart...")

models = ['Ridge\n(baseline)', 'CatBoost', 'XGBoost', 'LightGBM', 'Stacking\n(tuned)']
r2_values = [0.8274, 0.8910, 0.8978, 0.8963, 0.9026]
colors_bars = [COLORS['gray'], COLORS['primary'], COLORS['primary'],
               COLORS['primary'], '#1E40AF']

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

bars = ax.bar(models, r2_values, color=colors_bars, alpha=0.85,
              edgecolor='white', linewidth=0.5, width=0.6)

# Označi best model
bars[-1].set_edgecolor('#1E40AF')
bars[-1].set_linewidth(2)

# Vrijednosti iznad barova
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#111827')

# Referentna linija 0.90
ax.axhline(y=0.90, color=COLORS['green'], linestyle='--',
           linewidth=1.5, alpha=0.7, label='R² = 0.90')
ax.text(4.4, 0.901, 'R² = 0.90', color=COLORS['green'],
        fontsize=9, va='bottom')

ax.set_ylim(0.78, 0.92)
ax.set_ylabel('R² koeficijent determinacije', fontsize=11)
ax.set_title('Usporedba performansi modela\n(R² na test skupu)', fontsize=13)

# Anotacija poboljšanja
improvement = r2_values[-1] - r2_values[0]
ax.annotate(f'+{improvement*100:.1f}% poboljšanje',
            xy=(4, r2_values[-1]), xytext=(2.5, 0.915),
            arrowprops=dict(arrowstyle='->', color='#374151', lw=1.5),
            fontsize=10, color='#374151',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0FDF4',
                     edgecolor=COLORS['green'], alpha=0.9))

plt.tight_layout()
plt.savefig('viz_4_model_progression.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Spremljeno: viz_4_model_progression.png")

# ============================================================================
# 5. KORELACIJSKA MATRICA
# ============================================================================

print("\n5. Korelacijska matrica...")

key_features = [
    'price_eur',
    'm2', 'rooms', 'floor', 'total_floors',
    'year_built', 'building_age', 'effective_age',
    'energetski_razred_num',
    'num_bathrooms',
    'has_lift', 'has_parking', 'is_novogradnja',
    'floor_ratio', 'district_median_price_per_m2'
]

# Rename za ljepši prikaz
rename_map = {
    'price_eur': 'Cijena (€)',
    'm2': 'Površina (m²)',
    'rooms': 'Broj soba',
    'floor': 'Kat',
    'total_floors': 'Ukupno katova',
    'year_built': 'Godina gradnje',
    'building_age': 'Starost zgrade',
    'effective_age': 'Efektivna starost',
    'energetski_razred_num': 'Energetski razred',
    'num_bathrooms': 'Broj kupaonica',
    'has_lift': 'Lift',
    'has_parking': 'Parking',
    'is_novogradnja': 'Novogradnja',
    'floor_ratio': 'Pozicija u zgradi',
    'district_median_price_per_m2': 'Median cijena/m² (kvart)'
}

existing_features = [f for f in key_features if f in df.columns]
corr_df = df[existing_features].rename(columns=rename_map).corr()

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

mask = np.zeros_like(corr_df, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True

sns.heatmap(
    corr_df,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'shrink': 0.8, 'label': 'Pearsonov koeficijent korelacije'},
    annot_kws={'size': 8},
    ax=ax
)

ax.set_title('Korelacijska matrica ključnih značajki\n(Pearsonov koeficijent)', fontsize=13)
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
plt.savefig('viz_5_korelacijska_matrica.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Spremljeno: viz_5_korelacijska_matrica.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("  VIZUALIZACIJE ZAVRŠENE!")
print("="*60)
print("  viz_1_predicted_vs_actual.png")
print("  viz_2_feature_importance.png")
print("  viz_3_boxplot_kvartovi.png")
print("  viz_4_model_progression.png")
print("  viz_5_korelacijska_matrica.png")
print("="*60)