#!/usr/bin/env python3
"""
Njuškalo Data Cleaning v2
- 52 značajke + interakcijske značajke
- log(price) target, log(m2) feature
- Ispravno tretiranje sentinel vrijednosti (99, 999, -1)
- Sve kolumne iz oglasa iskorištene
"""

import pandas as pd
import numpy as np
import re
import json

def clean_njuskalo_data(input_file='njuskalo_apartments_partial.json', output_file='apartments_clean_v2.csv'):

    print("=" * 70)
    print("NJUŠKALO DATA CLEANING v2")
    print("=" * 70)

    # ========== LOAD ==========
    print("\n📂 Loading data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"   ✓ Loaded {len(df)} apartments, {len(df.columns)} columns")

    # ========== DEDUPLIKACIJA ==========
    original_count = len(df)
    df = df.drop_duplicates(subset=['oglas_id'], keep='first')
    print(f"\n🗑️  Duplicates removed: {original_count - len(df)}")

    # ========== 1. M2 ==========
    print("\n📏 Cleaning m2...")
    def clean_m2(value):
        if pd.isna(value):
            return None
        cleaned = str(value).replace(' m²', '').replace(',', '.')
        try:
            return float(cleaned)
        except:
            return None

    df['m2'] = df['stambena_površina'].apply(clean_m2)
    print(f"   ✓ Converted: {df['m2'].notna().sum()}/{len(df)}")

    # ========== 2. SOBE ==========
    print("\n🚪 Cleaning rooms...")
    def clean_rooms(value):
        if pd.isna(value):
            return None
        value_str = str(value).lower()
        if 'garsonijera' in value_str or 'studio' in value_str:
            return 1
        if '1-sobni' in value_str or 'jednosoban' in value_str:
            return 1
        if '2-sobni' in value_str or 'dvosoban' in value_str:
            return 2
        if '3-sobni' in value_str or 'trosoban' in value_str:
            return 3
        if '4-sobni' in value_str or 'četverosoban' in value_str:
            return 4
        if '5' in value_str:
            return 5
        match = re.search(r'\d+', value_str)
        return int(match.group()) if match else None

    df['rooms'] = df['broj_soba'].apply(clean_rooms)
    print(f"   ✓ Distribution: {df['rooms'].value_counts().sort_index().to_dict()}")

    # ========== 3. KAT ==========
    print("\n🏢 Cleaning floor...")
    def clean_floor(value):
        if pd.isna(value):
            return None
        value_str = str(value).lower().strip()
        if 'suteren' in value_str or 'basement' in value_str:
            return -1
        if 'visoko prizemlje' in value_str:
            return 0
        if 'prizemlje' in value_str:
            return 0
        if 'potkrovlje' in value_str or 'mansarda' in value_str or 'penthouse' in value_str:
            return 99  # sentinel, handled later
        value_str = value_str.replace('.', '').strip()
        match = re.search(r'\d+', value_str)
        return int(match.group()) if match else None

    df['floor_raw'] = df['kat'].apply(clean_floor)

    # Sentinel 99 → is_attic + median floor
    df['is_attic'] = (df['floor_raw'] == 99).astype(int)
    median_floor = df[df['floor_raw'] != 99]['floor_raw'].median()
    df['floor'] = df['floor_raw'].apply(lambda x: median_floor if x == 99 else x)
    df['floor'] = df['floor'].fillna(1)
    print(f"   ✓ is_attic: {df['is_attic'].sum()}, median floor used for attic: {median_floor:.0f}")

    # ========== 4. UKUPNI KATOVI ==========
    print("\n🏗️  Cleaning total floors...")
    def clean_total_floors(value):
        if pd.isna(value):
            return None
        value_str = str(value).lower()
        if 'prizemlje' in value_str:
            return 1
        match = re.search(r'\d+', value_str)
        return int(match.group()) if match else None

    df['total_floors'] = df['ukupni_broj_katova'].apply(clean_total_floors)
    df['total_floors'] = df.groupby('lokacija')['total_floors'].transform(
        lambda x: x.fillna(x.median())
    )
    df['total_floors'] = df['total_floors'].fillna(df['total_floors'].median())
    print(f"   ✓ Range: {df['total_floors'].min():.0f} - {df['total_floors'].max():.0f}")

    # ========== 5. GODINA IZGRADNJE ==========
    print("\n📅 Cleaning year built...")
    def clean_year(value):
        if pd.isna(value):
            return None
        value_str = str(value).replace('.', '').strip()
        match = re.search(r'\d{4}', value_str)
        if match:
            year = int(match.group())
            if 1850 <= year <= 2030:
                return year
        return None

    df['year_built'] = df['godina_izgradnje'].apply(clean_year)
    median_year = df['year_built'].median()
    df['year_built'] = df['year_built'].fillna(median_year)
    print(f"   ✓ Range: {df['year_built'].min():.0f} - {df['year_built'].max():.0f}")

    # ========== 6. GODINA RENOVACIJE ==========
    print("\n🔨 Cleaning renovation year...")
    def clean_renovation_year(value):
        if pd.isna(value):
            return None
        value_str = str(value).replace('.', '').strip()
        match = re.search(r'\d{4}', value_str)
        if match:
            year = int(match.group())
            if 1850 <= year <= 2030:
                return year
        return None

    df['renovation_year'] = df['godina_zadnje_renovacije'].apply(clean_renovation_year)
    # Sentinel: never_renovated flag + median imputation
    df['never_renovated'] = df['renovation_year'].isna().astype(int)
    current_year = 2026
    df['years_since_renovation'] = df['renovation_year'].apply(
        lambda x: current_year - x if pd.notna(x) else None
    )
    median_renovation = df['years_since_renovation'].median()
    df['years_since_renovation'] = df['years_since_renovation'].fillna(median_renovation)
    print(f"   ✓ Never renovated: {df['never_renovated'].sum()} ({df['never_renovated'].mean()*100:.1f}%)")

    # ========== 7. CIJENA ==========
    print("\n💰 Price column...")
    print(f"   ✓ Already clean: {df['price_eur'].notna().sum()}/{len(df)}")
    print(f"   ✓ Range: €{df['price_eur'].min():,.0f} - €{df['price_eur'].max():,.0f}")

    # ========== 8. UKLANJANJE OUTLIERA ==========
    print("\n🎯 Removing outliers...")
    before = len(df)
    df = df[df['price_eur'].notna()]
    df = df[df['m2'].notna()]
    df = df[(df['price_eur'] >= 30000) & (df['price_eur'] <= 2000000)]
    price_removed = before - len(df)

    before = len(df)
    df = df[(df['m2'] >= 15) & (df['m2'] <= 500)]
    m2_removed = before - len(df)

    df['price_per_m2'] = df['price_eur'] / df['m2']
    before = len(df)
    df = df[(df['price_per_m2'] >= 1000) & (df['price_per_m2'] <= 10000)]
    ppm2_removed = before - len(df)

    print(f"   ✗ Price: {price_removed}, m²: {m2_removed}, price/m²: {ppm2_removed}")
    print(f"   ✓ Remaining: {len(df)}")

    # ========== 9. ENERGETSKI RAZRED ==========
    print("\n⚡ Cleaning energy class...")
    energy_map = {'A+': 7, 'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}

    def clean_energy(value):
        if pd.isna(value):
            return None
        return energy_map.get(str(value).strip(), None)

    df['energetski_razred_num'] = df['energetski_razred'].apply(clean_energy)
    df['energy_unknown'] = df['energetski_razred_num'].isna().astype(int)
    median_energy = df['energetski_razred_num'].median()
    df['energetski_razred_num'] = df['energetski_razred_num'].fillna(median_energy)
    print(f"   ✓ Energy unknown: {df['energy_unknown'].sum()} ({df['energy_unknown'].mean()*100:.1f}%)")

    # ========== 10. BROJ KUPAONICA ==========
    print("\n🚿 Cleaning bathrooms...")
    def clean_bathrooms(value):
        if pd.isna(value):
            return None
        value_str = str(value).strip()
        if '5+' in value_str:
            return 5
        match = re.search(r'\d+', value_str)
        return int(match.group()) if match else None

    df['num_bathrooms'] = df['broj_kupaonica_s_wc-om'].apply(clean_bathrooms)
    df['num_bathrooms'] = df['num_bathrooms'].fillna(1)
    print(f"   ✓ Distribution: {df['num_bathrooms'].value_counts().sort_index().to_dict()}")

    # ========== 11. BROJ PARKING MJESTA ==========
    print("\n🅿️  Cleaning parking spots...")
    def clean_parking_count(value):
        if pd.isna(value):
            return 0
        value_str = str(value).lower()
        if 'nema' in value_str:
            return 0
        if '7+' in value_str:
            return 7
        match = re.search(r'\d+', value_str)
        return int(match.group()) if match else 0

    df['num_parking'] = df['broj_parkirnih_mjesta'].apply(clean_parking_count)
    print(f"   ✓ Distribution: {df['num_parking'].value_counts().sort_index().to_dict()}")

    # ========== 12. LOKACIJA ==========
    print("\n📍 Extracting location...")
    def extract_district(loc):
        if pd.isna(loc):
            return None
        parts = str(loc).split(',')
        if len(parts) >= 2:
            return parts[1].strip()
        return None

    df['district'] = df['lokacija'].apply(extract_district)

    district_counts = df['district'].value_counts()
    districts_to_keep = district_counts[district_counts >= 10].index.tolist()
    df['district_filtered'] = df['district'].apply(
        lambda x: x if x in districts_to_keep else 'Other'
    )

    district_dummies = pd.get_dummies(df['district_filtered'], prefix='district', drop_first=True)
    district_dummies = district_dummies.fillna(0)
    district_dummy_cols = district_dummies.columns.tolist()
    df = pd.concat([df, district_dummies], axis=1)
    print(f"   ✓ {len(districts_to_keep)} districts, {len(district_dummy_cols)} dummy vars")

    # ========== 13. BINARNE ZNAČAJKE ==========
    print("\n✨ Creating binary features...")

    # Novogradnja
    df['is_novogradnja'] = (
        df['title'].str.contains('novogradnja|nova gradnja', case=False, na=False) |
        df['podaci_o_objektu'].str.contains('Novogradnja', case=False, na=False)
    ).astype(int)

    # Parking
    df['has_parking'] = (
        (df['vrsta_parkinga'].notna()) |
        (df['broj_parkirnih_mjesta'].notna() & ~df['broj_parkirnih_mjesta'].str.contains('nema', case=False, na=False))
    ).astype(int)

    # Garaža (premium parking)
    df['has_garage'] = df['vrsta_parkinga'].str.contains('Garaža|Garažno mjesto', case=False, na=False).astype(int)

    # Balkon, lođa, terasa zasebno
    df['has_balcony'] = df['balkon_lođa_terasa'].str.contains('Balkon', case=False, na=False).astype(int)
    df['has_loggia'] = df['balkon_lođa_terasa'].str.contains('Lođa', case=False, na=False).astype(int)
    df['has_terrace'] = df['balkon_lođa_terasa'].str.contains('Terasa', case=False, na=False).astype(int)

    # Lift
    df['has_lift'] = df['podaci_o_objektu'].str.contains('Lift', case=False, na=False).astype(int)

    # Grijanje
    df['has_gas_heating'] = df['sustav_grijanja'].str.contains('plinsko|Peć na plin', case=False, na=False).astype(int)
    df['has_heat_pump'] = df['sustav_grijanja'].str.contains('Dizalica topline', case=False, na=False).astype(int)
    df['has_central_heating'] = df['sustav_grijanja'].str.contains('toplana|Zajednička kotlovnica', case=False, na=False).astype(int)

    # Klima
    df['has_ac'] = df['klima_uredaj'].notna().astype(int)

    # Podno grijanje
    df['has_floor_heating'] = df['funkcionalnosti_i_ostale_karakteristike'].str.contains('Podno grijanje', case=False, na=False).astype(int)

    # Protuprovalna vrata
    df['has_security_door'] = df['funkcionalnosti_i_ostale_karakteristike'].str.contains('Protuprovalna vrata', case=False, na=False).astype(int)

    # Video portafon
    df['has_video_intercom'] = df['funkcionalnosti_i_ostale_karakteristike'].str.contains('Video Portafon', case=False, na=False).astype(int)

    # Električne rolete
    df['has_electric_shutters'] = df['funkcionalnosti_i_ostale_karakteristike'].str.contains('Električne rolete', case=False, na=False).astype(int)

    # Alarm
    df['has_alarm'] = df['funkcionalnosti_i_ostale_karakteristike'].str.contains('Alarmni sustav', case=False, na=False).astype(int)

    # Luksuz
    df['has_luxury'] = df['funkcionalnosti_i_ostale_karakteristike'].str.contains('Jacuzzi|Sauna|Kamin', case=False, na=False).astype(int)

    # Ostali objekti
    df['has_storage'] = df['ostali_objekti_i_površine'].str.contains('Spremište|Podrum', case=False, na=False).astype(int)
    df['has_garden'] = df['ostali_objekti_i_površine'].str.contains('Dvorište|vrt', case=False, na=False).astype(int)
    df['has_pool'] = df['ostali_objekti_i_površine'].str.contains('Bazen', case=False, na=False).astype(int)

    # Dozvole
    df['has_ownership_doc'] = df['dozvole'].str.contains('Vlasnički list', case=False, na=False).astype(int)
    df['has_building_permit'] = df['dozvole'].str.contains('Građevinska dozvola', case=False, na=False).astype(int)
    df['has_usage_permit'] = df['dozvole'].str.contains('Uporabna dozvola', case=False, na=False).astype(int)

    # Namještenost
    df['is_furnished'] = df['namještenost_i_stanje'].str.contains('Potpuno namješten', case=False, na=False).astype(int)
    df['is_partially_furnished'] = df['namještenost_i_stanje'].str.contains('Djelomično namješten', case=False, na=False).astype(int)
    df['needs_renovation'] = df['namještenost_i_stanje'].str.contains('Za renoviranje|Roh-bau', case=False, na=False).astype(int)

    # Orijentacija
    df['has_south'] = df['orijentacija_stana'].str.contains('Jug', case=False, na=False).astype(int)
    df['has_north'] = df['orijentacija_stana'].str.contains('Sjever', case=False, na=False).astype(int)
    df['has_east'] = df['orijentacija_stana'].str.contains('Istok', case=False, na=False).astype(int)
    df['has_west'] = df['orijentacija_stana'].str.contains('Zapad', case=False, na=False).astype(int)

    # Alternativna energija
    df['has_alternative_energy'] = df['alternativni_izvori_energije'].notna().astype(int)

    # Duplex
    df['is_duplex'] = df['broj_etaža'].str.contains('Dvoetažni|Višeetažni', case=False, na=False).astype(int)

    print(f"   ✓ Created {sum(1 for c in df.columns if c.startswith('has_') or c.startswith('is_') or c.startswith('needs_'))} binary features")

    # ========== 14. ENGINEERED FEATURES ==========
    print("\n🎨 Creating engineered features...")

    current_year = 2026
    df['building_age'] = current_year - df['year_built']

    # floor_ratio (bez sentinel 99)
    df['floor_ratio'] = df.apply(
        lambda row: row['floor'] / (row['total_floors'] + 1)
        if pd.notna(row['total_floors']) and row['total_floors'] > 0 else 0,
        axis=1
    )

    # is_top_floor (bez sentinel)
    df['is_top_floor'] = (
        (df['floor'] == df['total_floors']) | (df['is_attic'] == 1)
    ).astype(int)

    # is_ground_or_lower
    df['is_ground_or_lower'] = (df['floor'] <= 0).astype(int)

    # rooms_per_m2 i m2_per_room
    df['rooms_per_m2'] = df['rooms'] / df['m2']
    df['m2_per_room'] = df['m2'] / df['rooms']

    # effective_age = min(building_age, years_since_renovation)
    df['effective_age'] = df.apply(
        lambda row: min(row['building_age'], row['years_since_renovation'])
        if row['never_renovated'] == 0 else row['building_age'],
        axis=1
    )

    # Veličinska kategorija (s overlappingom)
    df['is_small'] = (df['m2'] < 55).astype(int)
    df['is_medium'] = ((df['m2'] >= 45) & (df['m2'] < 100)).astype(int)
    df['is_large'] = (df['m2'] >= 85).astype(int)

    # log(m2)
    df['log_m2'] = np.log(df['m2'])

    # district median price per m2
    district_median_ppm2 = df.groupby('district')['price_per_m2'].transform('median')
    df['district_median_price_per_m2'] = district_median_ppm2

    print(f"   ✓ building_age: {df['building_age'].min():.0f} - {df['building_age'].max():.0f}")
    print(f"   ✓ Small: {df['is_small'].sum()}, Medium: {df['is_medium'].sum()}, Large: {df['is_large'].sum()}")

    # ========== 15. INTERAKCIJSKE ZNAČAJKE ==========
    print("\n🔗 Creating interaction features...")

    # Floor × Lift
    df['floor_x_lift'] = df['floor'] * df['has_lift']
    df['high_floor_no_lift'] = ((df['floor'] >= 3) & (df['has_lift'] == 0)).astype(int)
    df['bad_floor_access'] = ((df['floor'] >= 4) & (df['has_lift'] == 0)).astype(int)

    # Top floor × Terrace (penthouse efekt)
    df['top_floor_terrace'] = df['is_top_floor'] * df['has_terrace']

    # Top floor × Attic × Terrace
    df['attic_terrace'] = df['is_attic'] * df['has_terrace']

    # Ground floor × Garden
    df['ground_floor_garden'] = df['is_ground_or_lower'] * df['has_garden']

    # Building age × Renovation
    df['old_not_renovated'] = (
        (df['building_age'] > 50) & (df['years_since_renovation'] > 20)
    ).astype(int)

    # Novogradnja × Energetski razred
    df['new_building_energy'] = df['is_novogradnja'] * df['energetski_razred_num']

    # Luxury × Large apartment
    df['luxury_large'] = df['has_luxury'] * (df['m2'] > 120).astype(int)

    # South × No AC (ljeti problem)
    df['south_no_ac'] = df['has_south'] * (1 - df['has_ac'])

    # Terrace × High floor
    df['terrace_high_floor'] = df['has_terrace'] * df['floor']

    print(f"   ✓ Created 10 interaction features")

    # ========== 16. LOG TARGET ==========
    print("\n📊 Creating log price target...")
    df['log_price'] = np.log(df['price_eur'])
    print(f"   ✓ log_price range: {df['log_price'].min():.2f} - {df['log_price'].max():.2f}")

    # ========== 17. FINALNE KOLUMNE ==========
    print("\n📋 Selecting final columns...")

    final_columns = [
        # Identifikatori
        'oglas_id', 'url', 'title',

        # TARGET
        'price_eur', 'log_price', 'price_per_m2',

        # Glavne numeričke
        'm2', 'log_m2', 'rooms', 'floor', 'total_floors',
        'year_built', 'building_age', 'effective_age',
        'floor_ratio', 'rooms_per_m2', 'm2_per_room',
        'num_bathrooms', 'num_parking',
        'years_since_renovation', 'energetski_razred_num',
        'district_median_price_per_m2',

        # Flags za sentinel vrijednosti
        'is_attic', 'never_renovated', 'energy_unknown',

        # Lokacija
        'lokacija', 'district',

        # Veličinska kategorija
        'is_small', 'is_medium', 'is_large',

        # Binarne značajke
        'is_novogradnja', 'is_duplex',
        'has_parking', 'has_garage', 'num_parking',
        'has_balcony', 'has_loggia', 'has_terrace',
        'has_lift',
        'has_gas_heating', 'has_heat_pump', 'has_central_heating', 'has_electric_heating' if 'has_electric_heating' in df.columns else 'has_gas_heating',
        'has_ac', 'has_floor_heating',
        'has_security_door', 'has_video_intercom',
        'has_electric_shutters', 'has_alarm', 'has_luxury',
        'has_storage', 'has_garden', 'has_pool',
        'has_ownership_doc', 'has_building_permit', 'has_usage_permit',
        'is_furnished', 'is_partially_furnished', 'needs_renovation',
        'has_south', 'has_north', 'has_east', 'has_west',
        'has_alternative_energy',

        # Pozicijske
        'is_top_floor', 'is_ground_or_lower',

        # Interakcijske
        'floor_x_lift', 'high_floor_no_lift', 'bad_floor_access',
        'top_floor_terrace', 'attic_terrace',
        'ground_floor_garden',
        'old_not_renovated',
        'new_building_energy',
        'luxury_large',
        'south_no_ac',
        'terrace_high_floor',

        # Metadata
        'datum_objave', 'owner_name', 'scraped_at'
    ]

    # Dodaj district dummies
    final_columns.extend(district_dummy_cols)

    # Deduplikacija kolumni i provjera postojanja
    seen = set()
    final_columns_clean = []
    for col in final_columns:
        if col not in seen and col in df.columns:
            seen.add(col)
            final_columns_clean.append(col)

    df_clean = df[final_columns_clean].copy()

    print(f"   ✓ Final columns: {len(final_columns_clean)}")
    print(f"   ✓ Including {len(district_dummy_cols)} district dummies")

    # ========== 18. FINALNE STATISTIKE ==========
    print("\n" + "=" * 70)
    print("📊 FINAL DATASET STATISTICS")
    print("=" * 70)
    print(f"\n   Rows: {len(df_clean)}")
    print(f"   Columns: {len(df_clean.columns)}")
    print(f"   Missing values: {df_clean.isna().sum().sum()}")
    print(f"\n💰 Price:")
    print(f"   Min: €{df_clean['price_eur'].min():,.0f}")
    print(f"   Max: €{df_clean['price_eur'].max():,.0f}")
    print(f"   Mean: €{df_clean['price_eur'].mean():,.0f}")
    print(f"   Median: €{df_clean['price_eur'].median():,.0f}")
    print(f"\n📏 m2:")
    print(f"   Min: {df_clean['m2'].min():.0f}")
    print(f"   Max: {df_clean['m2'].max():.0f}")
    print(f"   Mean: {df_clean['m2'].mean():.0f}")
    print(f"\n📍 Top 5 districts:")
    print(df_clean['district'].value_counts().head(5))

    # ========== SAVE ==========
    print("\n💾 Saving...")
    df_clean.to_csv(output_file, index=False, encoding='utf-8')
    print(f"   ✓ Saved: {output_file} ({len(df_clean)} rows, {len(df_clean.columns)} columns)")

    print("\n" + "=" * 70)
    print("✅ CLEANING v2 COMPLETE!")
    print("=" * 70)

    return df_clean


if __name__ == "__main__":
    df = clean_njuskalo_data(
        input_file='njuskalo_apartments_partial.json',
        output_file='apartments_clean_v2.csv'
    )

    print(f"\n🎉 Done! {len(df)} apartments, {len(df.columns)} features")
    print("   Next: python train_models_v2.py")