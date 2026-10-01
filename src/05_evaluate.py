import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('apartments_clean_v2.csv')

with open('models_tuned.pkl', 'rb') as f:
    models = pickle.load(f)

with open('best_params.json', 'r') as f:
    best_params = json.load(f)

feature_cols = models['feature_cols']
X = df[feature_cols]
y = df['log_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print('Treniranje modela...')
xgb_m = xgb.XGBRegressor(**best_params['xgboost'], random_state=42, verbosity=0)
lgb_m = lgb.LGBMRegressor(**best_params['lightgbm'], random_state=42, verbose=-1)
cat_m = CatBoostRegressor(**best_params['catboost'], random_seed=42, verbose=0)

xgb_m.fit(X_train, y_train)
lgb_m.fit(X_train, y_train)
cat_m.fit(X_train, y_train)

# OOF za meta-model treniranje
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f'Fold {fold+1}/5...')
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr = y_train.iloc[tr_idx]

    m1 = xgb.XGBRegressor(**best_params['xgboost'], random_state=42, verbosity=0)
    m2 = lgb.LGBMRegressor(**best_params['lightgbm'], random_state=42, verbose=-1)
    m3 = CatBoostRegressor(**best_params['catboost'], random_seed=42, verbose=0)

    m1.fit(X_tr, y_tr)
    m2.fit(X_tr, y_tr)
    m3.fit(X_tr, y_tr)

    oof_xgb[val_idx] = m1.predict(X_val)
    oof_lgb[val_idx] = m2.predict(X_val)
    oof_cat[val_idx] = m3.predict(X_val)

# Meta-model
X_meta_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
meta_scaler = StandardScaler()
X_meta_train_s = meta_scaler.fit_transform(X_meta_train)
meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta_train_s, y_train)

# Test predikcije
pred_xgb = xgb_m.predict(X_test)
pred_lgb = lgb_m.predict(X_test)
pred_cat = cat_m.predict(X_test)

X_meta_test = np.column_stack([pred_xgb, pred_lgb, pred_cat])
X_meta_test_s = meta_scaler.transform(X_meta_test)
y_pred = meta_model.predict(X_meta_test_s)

r2 = r2_score(y_test, y_pred)
print(f'\nTest R2: {r2:.4f}')

price_pred = np.exp(y_pred)
price_actual = np.exp(y_test.values)
ratio = (price_pred - price_actual) / price_actual

def classify(r):
    if r >= 0.15:
        return 'GREEN'
    elif r <= -0.15:
        return 'RED'
    else:
        return 'YELLOW'

classes = [classify(r) for r in ratio]
print(f'GREEN: {classes.count("GREEN")}')
print(f'YELLOW: {classes.count("YELLOW")}')
print(f'RED: {classes.count("RED")}')

results = pd.DataFrame({
    'price_eur': price_actual,
    'predicted_price': price_pred,
    'classification': classes
})
results.to_csv('test_set_predictions.csv', index=False)
print('Saved: test_set_predictions.csv')