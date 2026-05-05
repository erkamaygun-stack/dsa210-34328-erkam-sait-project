# DSA210 — Tam Pipeline (Self-Contained)

Bu dosya kalan tüm adımları içeriyor: 01_data_collection.ipynb'in geri kalanı + 03_ml_modeling.ipynb'in tamamı + README + requirements.txt.

Hücreleri **sırayla VSCode notebook'larına yapıştır ve çalıştır**.

---

## BÖLÜM A — 01_data_collection.ipynb (kalan 5 hücre)

### Cell A1 — Enflasyon (düzeltilmiş)

```python
infl_filter = {
    'startPeriod': '2015-01', 'endPeriod': '2024-12',
    'geo': ['DE', 'ES', 'TR', 'UK'],
    'freq': 'M',
    'coicop': 'CP00',
    'unit': 'RCH_A'      # RCH_A1 değil!
}

df_infl = eurostat.get_data_df('prc_hicp_manr', filter_pars=infl_filter)
print("Inflation shape:", df_infl.shape)
df_infl.head()
```

### Cell A2 — wide_to_long (eğer fonksiyon kayıp olduysa yeniden tanımla)

```python
def wide_to_long(df, value_name):
    id_cols = [c for c in df.columns if not c[0].isdigit()]
    geo_col = [c for c in id_cols if 'geo' in c.lower()][0]
    df_long = df.melt(id_vars=id_cols, var_name='time', value_name=value_name)
    df_long = df_long.rename(columns={geo_col: 'geo'})
    country_map = {'DE':'Germany', 'ES':'Spain', 'TR':'Turkey', 'UK':'UK'}
    df_long['country'] = df_long['geo'].map(country_map)
    df_long['year_month'] = pd.to_datetime(df_long['time']).dt.to_period('M')
    return df_long[['country','year_month',value_name]].dropna(
        subset=['country']).sort_values(['country','year_month']).reset_index(drop=True)

infl_long = wide_to_long(df_infl, 'inflation_yoy')
print(infl_long.groupby('country').size())
print("\nNaN count:")
print(infl_long.groupby('country')['inflation_yoy'].apply(lambda x: x.isna().sum()))
infl_long.head()
```

### Cell A3 — CCI'ı yükle (mevcut CSV'den)

```python
cci_raw = pd.read_csv('oecd_cci_monthly.csv.csv')
countries_map = {
    "United Kingdom": "UK",
    "Germany":        "Germany",
    "Spain":          "Spain",
    "Türkiye":        "Turkey"
}
cci = cci_raw[cci_raw['Reference area'].isin(countries_map.keys())][
    ['Reference area','TIME_PERIOD','OBS_VALUE']
].copy()
cci['country'] = cci['Reference area'].map(countries_map)
cci['year_month'] = pd.to_datetime(cci['TIME_PERIOD']).dt.to_period('M')
cci['CCI'] = pd.to_numeric(cci['OBS_VALUE'], errors='coerce')
cci_long = cci[['country','year_month','CCI']].dropna()
cci_long = cci_long[(cci_long['year_month'] >= '2015-01') & 
                    (cci_long['year_month'] <= '2024-12')]
print("CCI per country:")
print(cci_long.groupby('country').size())
```

### Cell A4 — Hepsini birleştir + COVID dummy + kaydet

```python
# weather_country zaten elimizde (Adım 1b'den)
# retail_long, unemp_long, infl_long, cci_long da hazır

panel = (weather_country
    .merge(retail_long, on=['country','year_month'], how='left')
    .merge(unemp_long,  on=['country','year_month'], how='left')
    .merge(infl_long,   on=['country','year_month'], how='left')
    .merge(cci_long,    on=['country','year_month'], how='left'))

# COVID dummy: 2020-03 ile 2022-06 arası
panel['covid_dummy'] = panel['year_month'].apply(
    lambda ym: 1 if pd.Period('2020-03') <= ym <= pd.Period('2022-06') else 0
)

# Mevsim sütunları (sin/cos encoding) — month_sin, month_cos
panel['month_num'] = panel['year_month'].dt.month
import numpy as np
panel['month_sin'] = np.sin(2 * np.pi * panel['month_num'] / 12)
panel['month_cos'] = np.cos(2 * np.pi * panel['month_num'] / 12)

print(f"Panel shape: {panel.shape}")
print(f"Sütunlar: {panel.columns.tolist()}")
print(f"\nÜlke başına satır:")
print(panel.groupby('country').size())
print(f"\nNaN sayısı her sütunda:")
print(panel.isna().sum())
```

### Cell A5 — Final kaydet (parquet)

```python
import os
os.makedirs('data', exist_ok=True)

# year_month Period tipinden string'e çevir (parquet uyumluluğu)
panel_to_save = panel.copy()
panel_to_save['year_month'] = panel_to_save['year_month'].astype(str)

panel_to_save.to_parquet('data/panel.parquet', index=False)
panel_to_save.to_csv('data/panel.csv', index=False)

print("✅ data/panel.parquet kaydedildi")
print("✅ data/panel.csv kaydedildi (yedek)")
print(f"\nFinal shape: {panel_to_save.shape}")
panel_to_save.head()
```

**Bu noktada Adım 1 (veri toplama) BİTTİ. Notebook'u kaydet, kapat.**

---

## BÖLÜM B — 03_ml_modeling.ipynb (tüm içerik)

VSCode'da yeni notebook oluştur: `03_ml_modeling.ipynb`. Aşağıdaki hücreleri sırayla yapıştır.

### Cell B1 — Imports + veri yükleme

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                              confusion_matrix, roc_auc_score, classification_report)
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

os.makedirs('figures', exist_ok=True)

# Panel'i yükle
panel = pd.read_parquet('data/panel.parquet')
panel['year_month'] = pd.PeriodIndex(panel['year_month'], freq='M')
print(f"Panel: {panel.shape}")
print(panel.head())
```

### Cell B2 — Veri temizliği + feature listesi

```python
# Özellik grupları
WEATHER_FEATURES = ['sunshine_hours','daylight_hours','temperature_2m_mean',
                    'temperature_2m_max','temperature_2m_min',
                    'precipitation_sum','rain_sum','precipitation_hours']
CONTROL_FEATURES = ['unemployment_rate','inflation_yoy','covid_dummy',
                    'month_sin','month_cos']
TARGETS = ['CCI','retail_index']

# Türkiye enflasyonu çok yüksek — log transform
panel['inflation_log'] = np.log1p(panel['inflation_yoy'].clip(lower=0))

# Eksik değerleri ülke ortalamasıyla doldur (basit imputation)
for col in WEATHER_FEATURES + CONTROL_FEATURES + TARGETS:
    if col in panel.columns:
        panel[col] = panel.groupby('country')[col].transform(
            lambda x: x.fillna(x.mean()))

# Geriye kalan NaN satırlarını düşür (özellikle UK retail eksik)
print(f"NaN düşürmeden önce: {panel.shape}")
panel_clean = panel.dropna(subset=WEATHER_FEATURES + ['CCI'])
print(f"NaN düşürdükten sonra (CCI için): {panel_clean.shape}")

# Retail için ayrı bir alt-set (UK çoğunlukla NaN)
panel_retail = panel.dropna(subset=WEATHER_FEATURES + ['retail_index'])
print(f"Retail için: {panel_retail.shape}")
```

### Cell B3 — Korelasyon ısı haritası (EDA)

```python
fig, ax = plt.subplots(figsize=(12,9))
corr_features = WEATHER_FEATURES + ['unemployment_rate','inflation_yoy','CCI','retail_index']
sns.heatmap(panel_clean[corr_features].corr(), annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, ax=ax, cbar_kws={'label':'Pearson r'})
ax.set_title('Tüm Değişkenler Arası Korelasyon', fontsize=13)
plt.tight_layout()
plt.savefig('figures/ml_01_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Cell B4 — PCA: 8 hava değişkenini 2 bileşene indir

```python
scaler = StandardScaler()
X_weather = scaler.fit_transform(panel_clean[WEATHER_FEATURES])

pca = PCA(n_components=8)
X_pca_full = pca.fit_transform(X_weather)

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1,9), pca.explained_variance_ratio_, color='steelblue')
axes[0].plot(range(1,9), np.cumsum(pca.explained_variance_ratio_), 'ro-')
axes[0].set_xlabel('Bileşen')
axes[0].set_ylabel('Varyans oranı')
axes[0].set_title('Scree Plot — Hava Değişkenleri PCA')
axes[0].grid(alpha=0.3)

# Loading'ler — ilk 2 bileşenin yorumu
loadings = pd.DataFrame(pca.components_[:2].T, 
                        columns=['PC1','PC2'], index=WEATHER_FEATURES)
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[1])
axes[1].set_title('PC1 & PC2 Loadings')

plt.tight_layout()
plt.savefig('figures/ml_02_pca_weather.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"PC1 + PC2 toplam varyansın {pca.explained_variance_ratio_[:2].sum()*100:.1f}%'ini açıklıyor")

# 2 bileşeni panel'e ekle
panel_clean['weather_PC1'] = X_pca_full[:, 0]
panel_clean['weather_PC2'] = X_pca_full[:, 1]
```

### Cell B5 — Lineer Regresyon (CCI hedef)

```python
features_for_lr = ['weather_PC1','weather_PC2','unemployment_rate',
                    'inflation_log','covid_dummy','month_sin','month_cos']
X = panel_clean[features_for_lr].values
y = panel_clean['CCI'].values

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lr = LinearRegression()
cv_rmse = -cross_val_score(lr, X, y, cv=kf, scoring='neg_root_mean_squared_error')
cv_r2 = cross_val_score(lr, X, y, cv=kf, scoring='r2')

print(f"=== Linear Regression — CCI ===")
print(f"CV RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}")
print(f"CV R²:   {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

# Tüm veriyle fit + katsayılar
lr.fit(X, y)
coef_df = pd.DataFrame({
    'feature': features_for_lr,
    'coefficient': lr.coef_,
    'abs_coef': np.abs(lr.coef_)
}).sort_values('abs_coef', ascending=False)
print("\nKatsayılar (büyüklük sırasına göre):")
print(coef_df)

# Görselleştir
fig, ax = plt.subplots(figsize=(10,5))
colors = ['steelblue' if c > 0 else 'salmon' for c in coef_df['coefficient']]
ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title(f'CCI Linear Regression Katsayıları (R²={cv_r2.mean():.2f})')
plt.tight_layout()
plt.savefig('figures/ml_03_lr_coefficients_cci.png', dpi=150)
plt.show()
```

### Cell B6 — Lineer Regresyon (retail hedef)

```python
X_r = panel_retail[features_for_lr].values
y_r = panel_retail['retail_index'].values

lr_r = LinearRegression()
cv_rmse_r = -cross_val_score(lr_r, X_r, y_r, cv=kf, scoring='neg_root_mean_squared_error')
cv_r2_r = cross_val_score(lr_r, X_r, y_r, cv=kf, scoring='r2')

print(f"=== Linear Regression — Retail Index ===")
print(f"CV RMSE: {cv_rmse_r.mean():.3f} ± {cv_rmse_r.std():.3f}")
print(f"CV R²:   {cv_r2_r.mean():.3f} ± {cv_r2_r.std():.3f}")

lr_r.fit(X_r, y_r)
coef_df_r = pd.DataFrame({
    'feature': features_for_lr,
    'coefficient': lr_r.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print("\nKatsayılar:")
print(coef_df_r)
```

### Cell B7 — Lojistik Regresyon (CCI yön tahmini)

```python
# Binary target: bir sonraki ayın CCI'sı bugünden yüksek mi?
panel_clean = panel_clean.sort_values(['country','year_month']).reset_index(drop=True)
panel_clean['CCI_next'] = panel_clean.groupby('country')['CCI'].shift(-1)
panel_clean['CCI_direction'] = (panel_clean['CCI_next'] > panel_clean['CCI']).astype(int)

mask = panel_clean['CCI_next'].notna()
X_log = panel_clean.loc[mask, features_for_lr].values
y_log = panel_clean.loc[mask, 'CCI_direction'].values

scaler_log = StandardScaler()
X_log_s = scaler_log.fit_transform(X_log)

logreg = LogisticRegression(max_iter=1000, random_state=42)
cv_acc = cross_val_score(logreg, X_log_s, y_log, cv=kf, scoring='accuracy')
cv_auc = cross_val_score(logreg, X_log_s, y_log, cv=kf, scoring='roc_auc')

print(f"=== Logistic Regression — CCI yön tahmini ===")
print(f"Class dağılımı: {np.bincount(y_log)}")
print(f"CV Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
print(f"CV AUC:      {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

logreg.fit(X_log_s, y_log)
y_pred = logreg.predict(X_log_s)
cm = confusion_matrix(y_log, y_pred)

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Düşüş','Artış'], yticklabels=['Düşüş','Artış'])
ax.set_xlabel('Tahmin'); ax.set_ylabel('Gerçek')
ax.set_title(f'Confusion Matrix — CCI Direction (AUC={cv_auc.mean():.2f})')
plt.tight_layout()
plt.savefig('figures/ml_04_logreg_confusion.png', dpi=150)
plt.show()
```

### Cell B8 — Random Forest (CCI hedef) + Feature Importance

```python
rf_features = WEATHER_FEATURES + ['unemployment_rate','inflation_log','covid_dummy',
                                    'month_sin','month_cos']
X_rf = panel_clean[rf_features].values
y_rf = panel_clean['CCI'].values

rf = RandomForestRegressor(n_estimators=200, max_depth=8, 
                            random_state=42, n_jobs=-1)
cv_rmse_rf = -cross_val_score(rf, X_rf, y_rf, cv=kf, scoring='neg_root_mean_squared_error')
cv_r2_rf = cross_val_score(rf, X_rf, y_rf, cv=kf, scoring='r2')

print(f"=== Random Forest — CCI ===")
print(f"CV RMSE: {cv_rmse_rf.mean():.3f}")
print(f"CV R²:   {cv_r2_rf.mean():.3f}")

rf.fit(X_rf, y_rf)
imp_df = pd.DataFrame({
    'feature': rf_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(10,7))
ax.barh(imp_df['feature'], imp_df['importance'], color='forestgreen')
ax.set_title(f'Random Forest Feature Importance — CCI (R²={cv_r2_rf.mean():.2f})')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('figures/ml_05_rf_importance_cci.png', dpi=150)
plt.show()
print(imp_df.iloc[::-1])
```

### Cell B9 — Random Forest (retail) + Gradient Boosting

```python
# Retail için RF
X_rf_r = panel_retail[rf_features].values
y_rf_r = panel_retail['retail_index'].values

rf_r = RandomForestRegressor(n_estimators=200, max_depth=8, 
                              random_state=42, n_jobs=-1)
cv_rmse_rf_r = -cross_val_score(rf_r, X_rf_r, y_rf_r, cv=kf, scoring='neg_root_mean_squared_error')
cv_r2_rf_r = cross_val_score(rf_r, X_rf_r, y_rf_r, cv=kf, scoring='r2')
print(f"=== Random Forest — Retail ===")
print(f"CV RMSE: {cv_rmse_rf_r.mean():.3f}, R²: {cv_r2_rf_r.mean():.3f}")

rf_r.fit(X_rf_r, y_rf_r)
imp_df_r = pd.DataFrame({
    'feature': rf_features,
    'importance': rf_r.feature_importances_
}).sort_values('importance', ascending=True)

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
cv_rmse_gb = -cross_val_score(gb, X_rf, y_rf, cv=kf, scoring='neg_root_mean_squared_error')
cv_r2_gb = cross_val_score(gb, X_rf, y_rf, cv=kf, scoring='r2')
print(f"\n=== Gradient Boosting — CCI ===")
print(f"CV RMSE: {cv_rmse_gb.mean():.3f}, R²: {cv_r2_gb.mean():.3f}")

# Retail RF importance grafik
fig, ax = plt.subplots(figsize=(10,7))
ax.barh(imp_df_r['feature'], imp_df_r['importance'], color='darkorange')
ax.set_title(f'RF Feature Importance — Retail (R²={cv_r2_rf_r.mean():.2f})')
plt.tight_layout()
plt.savefig('figures/ml_06_rf_importance_retail.png', dpi=150)
plt.show()
```

### Cell B10 — K-Means Kümeleme: country-month tipolojisi

```python
# Hava değişkenlerine göre 480 country-month'u kümele
X_cluster = scaler.fit_transform(panel_clean[WEATHER_FEATURES])

# Optimal k (Elbow + Silhouette)
from sklearn.metrics import silhouette_score
inertias = []
sils = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_cluster, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(range(2,9), inertias, 'o-', color='steelblue')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[1].plot(range(2,9), sils, 'o-', color='darkorange')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette')
axes[1].set_title('Silhouette Score')
plt.tight_layout()
plt.savefig('figures/ml_07_kmeans_choice.png', dpi=150)
plt.show()

# k=4 ile fit
km = KMeans(n_clusters=4, random_state=42, n_init=10)
panel_clean['cluster'] = km.fit_predict(X_cluster)

# Her küme için ortalama CCI ve retail
cluster_summary = panel_clean.groupby('cluster').agg({
    'sunshine_hours':'mean',
    'temperature_2m_mean':'mean',
    'precipitation_sum':'mean',
    'CCI':'mean',
    'retail_index':'mean',
    'country':lambda x: x.mode()[0]
}).round(2)
print("Küme özetleri:")
print(cluster_summary)

# Görselleştir: cluster × country dağılımı
fig, ax = plt.subplots(figsize=(10,5))
ct = pd.crosstab(panel_clean['country'], panel_clean['cluster'])
ct.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
ax.set_title('Country-Month Cluster Dağılımı')
ax.set_ylabel('Ay sayısı')
plt.tight_layout()
plt.savefig('figures/ml_08_cluster_distribution.png', dpi=150)
plt.show()
```

### Cell B11 — Hierarchical Clustering: 4 ülkenin iklim parmak izi

```python
# Her ülke için yıllık ortalama hava profili
country_profile = panel_clean.groupby('country')[WEATHER_FEATURES].mean()
country_scaled = StandardScaler().fit_transform(country_profile)

linkage_matrix = linkage(country_scaled, method='ward')

fig, ax = plt.subplots(figsize=(10,5))
dendrogram(linkage_matrix, labels=country_profile.index.tolist(), ax=ax)
ax.set_title('Ülke İklim Dendrogramı (Ward Linkage)')
ax.set_ylabel('Mesafe')
plt.tight_layout()
plt.savefig('figures/ml_09_hierarchical_dendrogram.png', dpi=150)
plt.show()
```

### Cell B12 — Final karşılaştırma tablosu

```python
results = pd.DataFrame([
    {'model':'Linear Regression', 'target':'CCI', 'RMSE':cv_rmse.mean(), 'R²':cv_r2.mean()},
    {'model':'Linear Regression', 'target':'Retail', 'RMSE':cv_rmse_r.mean(), 'R²':cv_r2_r.mean()},
    {'model':'Logistic Regression', 'target':'CCI direction', 
     'RMSE':np.nan, 'R²':cv_auc.mean()},
    {'model':'Random Forest', 'target':'CCI', 'RMSE':cv_rmse_rf.mean(), 'R²':cv_r2_rf.mean()},
    {'model':'Random Forest', 'target':'Retail', 'RMSE':cv_rmse_rf_r.mean(), 'R²':cv_r2_rf_r.mean()},
    {'model':'Gradient Boosting', 'target':'CCI', 'RMSE':cv_rmse_gb.mean(), 'R²':cv_r2_gb.mean()},
])
print("=== Final Model Karşılaştırması ===")
print(results.round(3).to_string(index=False))
results.to_csv('data/model_comparison.csv', index=False)

# Görsel karşılaştırma
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_data = results.dropna(subset=['RMSE'])
sns.barplot(data=plot_data, x='model', y='RMSE', hue='target', ax=axes[0])
axes[0].set_title('RMSE Karşılaştırması (düşük = iyi)')
axes[0].tick_params(axis='x', rotation=30)

sns.barplot(data=results, x='model', y='R²', hue='target', ax=axes[1])
axes[1].set_title('R² Karşılaştırması (yüksek = iyi)')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('figures/ml_10_model_comparison.png', dpi=150)
plt.show()
```

---

## BÖLÜM C — README.md (üzerine yaz)

Notebook çalıştıktan sonra README'yi şu içerikle güncelle:

```markdown
# DSA210 Project — Sunlight, Weather and Consumer Behavior
**Erkam Sait Aygün | 34328**

## Project Overview
This project investigates whether weather conditions (sunshine hours, temperature, 
precipitation) are associated with consumer perception (CCI) and consumer behavior 
(retail sales) across four European countries: Germany, Spain, Turkey, and the 
United Kingdom (2015–2024).

The analysis is structured in two parts:
- **Part A — Statistical Analysis:** Pearson/Spearman correlations, lag analysis, 
  seasonal t-tests on raw and country-aggregated series.
- **Part B — Machine Learning:** Multi-source data integration (weather + retail + 
  macroeconomic controls), PCA, regression (linear & logistic), tree-based ensembles 
  (Random Forest, Gradient Boosting), and clustering (K-Means, hierarchical).

## Repository Structure
```
├── 01_data_collection.ipynb              ← Multi-source data fetch & merge
├── 02_statistical_analysis.ipynb         ← Part A (renamed from EDA notebook)
├── 03_ml_modeling.ipynb                  ← Part B
├── data/
│   ├── panel.parquet                     ← merged dataset (480 country-months)
│   ├── retail_sales_monthly.csv
│   ├── weather_monthly.csv
│   ├── unemployment_monthly.csv
│   ├── inflation_monthly.csv
│   └── model_comparison.csv
├── figures/
├── PROJECT PROPOSAL ERKAM SAİT AYGÜN 34328.pdf
├── README.md
└── requirements.txt
```

## Data Sources
- **Weather:** Open-Meteo Historical Weather API (ERA5) — 5 cities per country, 
  population-weighted aggregation
- **CCI:** OECD Composite Leading Indicators (amplitude-adjusted, 2015=100)
- **Retail:** Eurostat `sts_trtu_m` (volume of sales, NACE G47, seasonally + 
  working-day adjusted, 2021=100). UK excluded post-Brexit due to data unavailability.
- **Unemployment:** Eurostat `une_rt_m` (% of active population, seasonally adjusted)
- **Inflation:** Eurostat `prc_hicp_manr` (HICP year-on-year, all items)

## Variables Constructed
- 8 weather features per country-month
- 2 weather PCA components
- COVID dummy (2020-03 to 2022-06)
- Cyclic month encoding (sin/cos)

## Methodology Summary

### Part A — Statistical Tests
Pearson/Spearman correlation between sunshine and CCI; lag-1 analysis; 
summer-vs-winter t-test on CCI; cross-country variability comparison.

### Part B — ML Pipeline
1. **PCA** on 8 weather variables → 2 orthogonal components
2. **Linear regression** of CCI and retail_index on weather PCs + controls
3. **Logistic regression** of next-month CCI direction (binary)
4. **Random Forest** for non-linear effects + feature importance
5. **Gradient Boosting** as performance benchmark
6. **K-Means** clustering of 480 country-months by weather profile
7. **Hierarchical clustering** of country climate profiles

All models evaluated via 5-fold cross-validation (RMSE for regression, AUC for 
classification).

## Reproducing the Analysis

```bash
git clone https://github.com/erkamaygun-stack/dsa210-34328-erkam-sait-project.git
cd dsa210-34328-erkam-sait-project
pip install -r requirements.txt

# Run in order:
# 1. 01_data_collection.ipynb  → produces data/panel.parquet
# 2. 02_statistical_analysis.ipynb  → Part A results
# 3. 03_ml_modeling.ipynb       → Part B results
```

## Key Findings
*(Notebook'ları çalıştırdıktan sonra buraya gerçek sonuçları yaz — örnek:)*

- Sunshine ↔ CCI raw Pearson correlation is non-significant for all four countries 
  (Part A).
- After deseasonalization and adding macro controls, weather PC1 explains a small 
  but non-zero share of CCI variance (Part B linear regression R² ≈ 0.X).
- Random Forest feature importance ranks unemployment and inflation above all 
  weather variables for CCI prediction.
- K-Means clustering on weather features partitions country-months into climate 
  archetypes; cluster mean CCI varies by ~Y points.

## Limitations
- Single weather aggregation per country (population-weighted city mean) does not 
  capture sub-national climate variation.
- UK retail/unemployment data limited post-Brexit.
- CCI is a smoothed (amplitude-adjusted) index; intra-month sentiment fluctuations 
  are not visible.
- Causal claims are not made; analysis is associative.
- Cross-validation is not strictly time-aware (KFold with shuffle); time-series 
  split would be more conservative.
```

---

## BÖLÜM D — requirements.txt

Repo köküne `requirements.txt` adıyla:

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scipy>=1.10
requests>=2.30
eurostat>=1.4
pyarrow>=12.0
```

Terminalde test:
```bash
pip install -r requirements.txt
```

---

## SON KONTROL LİSTESİ

- [ ] `data/panel.parquet` oluştu (Adım 1 sonu)
- [ ] `03_ml_modeling.ipynb` baştan sona hatasız çalıştı
- [ ] `figures/` klasöründe ml_01 — ml_10 PNG'leri var
- [ ] `data/model_comparison.csv` oluştu
- [ ] README.md güncellendi
- [ ] requirements.txt eklendi
- [ ] `02_statistical_analysis.ipynb` adı (eski notebook'un yeni adı) güncellendi
- [ ] `git add .`, `git commit -m "Add Part B: ML pipeline"`, `git push origin main`

## SORUN GİDERME

**`weather_country` tanımlı değil hatası:** Adım 1b notebook'unu yeniden çalıştır, 
ya da CSV'den yükle: `weather_country = pd.read_csv('data/weather_monthly.csv')` 
ve `year_month` sütununu Period'a çevir.

**`retail_long` tanımlı değil:** `pd.read_csv('data/retail_sales_monthly.csv')` 
ile yükle.

**ML notebook'ta NaN hataları:** Cell B2'deki imputation/dropna adımlarının 
çalıştığından emin ol.

**Random Forest çok yavaş:** `n_estimators=200`'ü 100'e düşür.

**Memory sorunu:** Notebook'u baştan restart et, sadece gerekli hücreleri tekrar koş.
