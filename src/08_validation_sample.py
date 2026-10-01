#!/usr/bin/env python3
"""
Izbor 30 stanova za validaciju s Regentom v3
- Samo stanovi s ORIGINALNIM podacima (nije imputirano)
- Energetski razred i godina izgradnje moraju biti iz oglasa
- 10 GREEN, 10 YELLOW, 10 RED
"""

import pandas as pd
import numpy as np
import json

RAW_FILE = 'njuskalo_apartments_partial.json'
CLASSIFIED_FILE = 'apartments_classified_v2.csv'
OUTPUT_FILE = 'validacija_regent_30stanova_v3.csv'

# ========== UČITAJ SIROVE PODATKE ==========
print("Učitavam sirove podatke...")
with open(RAW_FILE, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

raw_df = pd.DataFrame(raw_data)

# Stanovi koji imaju ORIGINALNE vrijednosti (nije imputirano)
raw_complete = raw_df[
    raw_df['godina_izgradnje'].notna() &
    raw_df['energetski_razred'].notna() &
    raw_df['stambena_površina'].notna() &
    raw_df['broj_soba'].notna() &
    raw_df['price_eur'].notna()
]['oglas_id'].astype(str).tolist()

print(f"Stanova s originalnim podacima: {len(raw_complete)}")

# ========== UČITAJ KLASIFICIRANE PODATKE ==========
classified = pd.read_csv(CLASSIFIED_FILE, encoding='utf-8')
classified['oglas_id'] = classified['oglas_id'].astype(str)

# Filtriraj samo stanove s originalnim podacima
complete = classified[classified['oglas_id'].isin(raw_complete)].copy()
print(f"Nakon filtera u klasificiranom datasetu: {len(complete)}")
print(f"\nDistribucija:")
print(complete['classification'].value_counts())

# ========== ENERGETSKI RAZRED KAO STRING ==========
energy_map = {7: 'A+', 6: 'A', 5: 'B', 4: 'C', 3: 'D', 2: 'E', 1: 'F', 0: 'G'}
complete['energetski_razred_str'] = complete['energetski_razred_num'].round().astype(int).map(energy_map)

# ========== STANJE NEKRETNINE ==========
def get_stanje(row):
    if row.get('is_novogradnja', 0) == 1:
        return 'Novogradnja'
    elif row.get('needs_renovation', 0) == 1:
        return 'Potrebna renovacija'
    else:
        return 'Dobro stanje'

complete['stanje_nekretnine'] = complete.apply(get_stanje, axis=1)

# ========== 10 GREEN ==========
green = complete[complete['classification'] == 'GREEN'].sort_values('value_ratio_pct', ascending=False)
green_filtered = green[
    (green['value_ratio_pct'] >= 15) &
    (green['value_ratio_pct'] <= 40)
].iloc[5:55]
green_sample = green_filtered.sample(n=min(10, len(green_filtered)), random_state=42)
print(f"\nGREEN pool: {len(green_filtered)}, odabrano: {len(green_sample)}")

# ========== 10 YELLOW ==========
yellow = complete[complete['classification'] == 'YELLOW']
yellow_filtered = yellow[
    (yellow['value_ratio_pct'] >= -5) &
    (yellow['value_ratio_pct'] <= 5)
]
yellow_sample = yellow_filtered.sample(n=min(10, len(yellow_filtered)), random_state=42)
print(f"YELLOW pool: {len(yellow_filtered)}, odabrano: {len(yellow_sample)}")

# ========== 10 RED ==========
red = complete[complete['classification'] == 'RED'].sort_values('value_ratio_pct', ascending=True)
red_filtered = red[
    (red['value_ratio_pct'] <= -15) &
    (red['value_ratio_pct'] >= -40)
].iloc[5:55]
red_sample = red_filtered.sample(n=min(10, len(red_filtered)), random_state=42)
print(f"RED pool: {len(red_filtered)}, odabrano: {len(red_sample)}")

# ========== SPOJI ==========
validation_set = pd.concat([green_sample, yellow_sample, red_sample])
validation_set = validation_set.reset_index(drop=True)

# ========== OUTPUT ==========
output_cols = [
    'classification', 'value_ratio_pct',
    'price_eur', 'predicted_price', 'price_difference',
    'lokacija', 'district',
    'm2', 'rooms', 'num_bathrooms', 'year_built',
    'energetski_razred_str', 'stanje_nekretnine',
    'floor', 'is_novogradnja',
    'has_parking', 'has_garage', 'has_garden',
    'has_balcony', 'has_terrace', 'has_ac',
    'has_lift', 'has_pool', 'is_furnished',
    'url', 'oglas_id', 'datum_objave'
]

output_cols = [c for c in output_cols if c in validation_set.columns]
output = validation_set[output_cols].copy()

output['regent_procjena'] = ''
output['regent_raspon_min'] = ''
output['regent_raspon_max'] = ''
output['napomena'] = ''

output['sort_order'] = output['classification'].map({'GREEN': 0, 'YELLOW': 1, 'RED': 2})
output = output.sort_values(['sort_order', 'value_ratio_pct'], ascending=[True, False])
output = output.drop('sort_order', axis=1)
output = output.reset_index(drop=True)
output.index = output.index + 1

# ========== ISPIS ==========
print("\n" + "="*80)
print("30 STANOVA ZA VALIDACIJU — SAMO ORIGINALNI PODACI")
print("="*80)

for cls in ['GREEN', 'YELLOW', 'RED']:
    subset = output[output['classification'] == cls]
    print(f"\n{'='*40}")
    print(f"  {cls} ({len(subset)} stanova)")
    print(f"{'='*40}")
    for i, (_, row) in enumerate(subset.iterrows(), 1):
        print(f"\n  {i}. {row['lokacija']}")
        print(f"     Cijena:     €{row['price_eur']:,.0f}")
        print(f"     Predviđeno: €{row['predicted_price']:,.0f}")
        print(f"     Ratio:      {row['value_ratio_pct']:+.1f}%")
        print(f"     m²: {row['m2']:.0f}  Sobe: {row['rooms']:.0f}  Kat: {row['floor']:.0f}")
        print(f"     Godina: {row['year_built']:.0f}  Energetski: {row['energetski_razred_str']}  Stanje: {row['stanje_nekretnine']}")
        print(f"     Lift: {int(row['has_lift'])}  Parking: {int(row['has_parking'])}  Balkon: {int(row['has_balcony'])}  Terasa: {int(row['has_terrace'])}")
        print(f"     URL: {row['url']}")

output.to_csv(OUTPUT_FILE, index=True, encoding='utf-8')
print(f"\n{'='*80}")
print(f"✓ Spremljeno: {OUTPUT_FILE}")
print(f"✓ Svi podaci su ORIGINALNI iz oglasa, ništa nije imputirano")
print(f"{'='*80}\n")