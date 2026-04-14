# DSA210 Project — Sunlight Duration and Consumer Confidence
**Erkam Sait Aygün | 34328**

## Project Overview
This project investigates whether monthly sunlight duration is associated 
with changes in the Consumer Confidence Index (CCI) across four countries: 
Germany, Spain, Turkey, and the United Kingdom. The analysis covers the 
period 2015–2024.

## Repository Structure
- `DSA210_ErkamSaitAygun_34328_EDA_and_Hypothesis_Tests.ipynb` — Main notebook (data collection, EDA, hypothesis tests)
- `oecd_cci_monthly.csv` — Monthly CCI data from OECD Data Explorer
- `sunlight_monthly_4countries.csv` — Monthly sunlight hours from Open-Meteo API
- `PROJECT PROPOSAL ERKAM SAİT AYGÜN 34328` — Project proposal document

## Data Sources
- **CCI:** [OECD Data Explorer](https://data-explorer.oecd.org) — Composite Leading Indicators, amplitude-adjusted, monthly
- **Sunlight:** [Open-Meteo Historical Weather API](https://open-meteo.com) — Hourly sunshine duration aggregated to monthly totals

## How to Reproduce
1. Clone the repository
2. Upload `oecd_cci_monthly.csv` to your environment
3. Open the notebook in Google Colab or Jupyter
4. Run all cells in order

## Analysis Summary
- **EDA:** Time series plots, distribution histograms, heatmaps
- **Test 1:** Pearson & Spearman correlation — sunlight vs CCI
- **Test 2:** Lag analysis — previous month sunlight vs current CCI
- **Test 3:** Seasonal t-test — summer vs winter CCI
- **Test 4:** Country-level variability comparison
