# DSA210 Project — Sunlight, Weather and Consumer Behavior
**Erkam Sait Aygün | 34328**

## Project Overview
This project investigates whether weather conditions (sunshine, temperature,
precipitation) are associated with consumer perception (Consumer Confidence Index, CCI)
and consumer behavior (retail sales volume) across four European countries:
Germany, Spain, Turkey, and the United Kingdom (2015–2024).

The analysis is structured in two parts:
- **Part A — Statistical Analysis:** Pearson/Spearman correlations, lag analysis,
  and seasonal t-tests on raw country-level series.
- **Part B — Machine Learning:** Multi-source data integration (weather + retail +
  macroeconomic controls), PCA, regression (linear & logistic), tree-based ensembles
  (Random Forest, Gradient Boosting), and clustering (K-Means, hierarchical).

## Repository Structure
```
├── 01_data_collection.ipynb                              ← Multi-source data fetch & merge
├── DSA210_ErkamSaitAygun_34328_EDA_and_Hypothesis_Tests.ipynb  ← Part A
├── 03_ml_modeling.ipynb                                  ← Part B
├── data/
│   ├── panel.parquet                                     ← merged dataset (480 country-months)
│   ├── retail_sales_monthly.csv
│   ├── weather_monthly.csv
│   └── model_comparison.csv
├── figures/                                              ← all generated plots
├── PROJECT PROPOSAL ERKAM SAİT AYGÜN 34328.pdf
├── README.md
└── requirements.txt
```

## Data Sources
- **Weather:** Open-Meteo Historical Weather API — 5 cities per country, population-
  weighted aggregation. 8 variables (sunshine, daylight, mean/max/min temperature,
  precipitation, rain, precipitation hours).
- **CCI:** OECD Composite Leading Indicators (amplitude-adjusted).
- **Retail Sales:** Eurostat `sts_trtu_m` (volume of sales, NACE G47, seasonally +
  working-day adjusted, base 2021=100). UK excluded post-Brexit due to data unavailability.
- **Unemployment:** Eurostat `une_rt_m` (% of active population, seasonally adjusted).
- **Inflation:** Eurostat `prc_hicp_manr` (HICP year-on-year, all items).
- **COVID dummy:** 2020-03 to 2022-06.

## Methodology Summary

### Part A — Statistical Tests
Pearson/Spearman correlation between sunshine and CCI; lag-1 analysis; summer-vs-winter
t-test on CCI; cross-country variability comparison. All four countries show
non-significant raw correlations (p > 0.05).

### Part B — ML Pipeline
1. **PCA** on 8 weather variables → 2 orthogonal components (weather_PC1, weather_PC2)
2. **Linear regression** of CCI and retail_index on weather PCs + macro controls
3. **Logistic regression** of next-month CCI direction (binary)
4. **Random Forest** for non-linear effects + feature importance
5. **Gradient Boosting** as performance benchmark
6. **K-Means** clustering of 480 country-months by weather profile
7. **Hierarchical clustering** of country-level climate profiles (dendrogram)

All regression models evaluated via 5-fold cross-validation; RMSE for regression,
AUC for classification.

## Key Findings

| Model | Target | R² (CV) | RMSE (CV) |
|-------|--------|---------|-----------|
| Linear Regression | CCI | 0.021 | 2.36 |
| Linear Regression | Retail Index | 0.135 | 14.89 |
| Logistic Regression | CCI direction | AUC 0.469 | — |
| **Random Forest** | **CCI** | **0.324** | **1.97** |
| **Random Forest** | **Retail Index** | **0.871** | **5.51** |
| Gradient Boosting | CCI | 0.312 | 1.99 |

Three observations:

1. **Non-linear models add substantial value.** Random Forest improves CCI prediction
   by 15× over Linear Regression (R² 0.02 → 0.32) and retail prediction by 6.4×
   (R² 0.14 → 0.87). The non-significant Pearson correlations in Part A do not
   imply absence of association; they imply that the association is non-linear and
   confounded with seasonality.
2. **Weather + macro controls explain consumer behavior much better than consumer
   perception.** Maximum R² for retail (0.87) is more than twice the maximum R² for
   CCI (0.32). The amplitude-adjusted CCI is highly smoothed and robust to weather
   shocks; observed retail volume is not.
3. **Direction of monthly CCI change is essentially unpredictable** from weather +
   macro features (logistic regression AUC = 0.47, below random).

The Random Forest retail model's high R² should not be interpreted as a pure
weather effect. Feature importance analysis (`figures/ml_06_rf_importance_retail.png`)
indicates that inflation and unemployment dominate the predictive signal, with
weather contributing a smaller but non-zero share.

## Reproducing the Analysis

```bash
git clone https://github.com/erkamaygun-stack/dsa210-34328-erkam-sait-project.git
cd dsa210-34328-erkam-sait-project
pip install -r requirements.txt

# Run notebooks in order:
# 1. 01_data_collection.ipynb           → produces data/panel.parquet
# 2. DSA210_ErkamSaitAygun_34328_EDA_and_Hypothesis_Tests.ipynb  → Part A results
# 3. 03_ml_modeling.ipynb               → Part B results + figures
```

## Limitations
- Weather is aggregated at the country level via 5-city population-weighted means;
  sub-national climate variation (especially in Spain and Turkey) is not captured.
- UK retail and unemployment data discontinued post-Brexit; UK is partially excluded
  from retail-target analyses.
- CCI is amplitude-adjusted by OECD, which dampens intra-month variability;
  short-horizon weather effects on perception are likely not detectable.
- Cross-validation uses K-Fold with shuffling; a strict time-series split would be
  more conservative for inferential claims.
- Causal claims are not made; analysis is associative only.
