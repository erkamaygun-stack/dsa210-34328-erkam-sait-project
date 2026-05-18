from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

app = Flask(__name__)

COUNTRIES = ["Germany", "Spain", "Turkey", "UK"]
COUNTRY_COLORS = {
    "Germany": "#2563eb",
    "Spain": "#dc2626",
    "Turkey": "#16a34a",
    "UK": "#7c3aed",
}

WEATHER_FEATURES = [
    "sunshine_hours",
    "daylight_hours",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
]

CONTROL_FEATURES = [
    "unemployment_rate",
    "inflation_yoy",
    "covid_dummy",
    "month_sin",
    "month_cos",
]

NAV_ITEMS = [
    ("home", "Home", "fa-house"),
    ("motivation", "Motivation", "fa-compass"),
    ("data_sources", "Data Sources", "fa-database"),
    ("methodology", "Methodology", "fa-gears"),
    ("eda", "EDA", "fa-chart-area"),
    ("findings", "Findings", "fa-flask"),
    ("ml_results", "ML Results", "fa-robot"),
    ("explorer", "Explorer", "fa-sliders"),
    ("conclusions", "Conclusions", "fa-flag-checkered"),
]

DATA_DICTIONARY = [
    ("country", "string", "Country label: Germany, Spain, Turkey, UK"),
    ("year_month", "date", "Monthly observation period from 2015-01 to 2024-12"),
    ("sunshine_hours", "float", "Population-weighted monthly sunshine hours"),
    ("daylight_hours", "float", "Population-weighted monthly daylight hours"),
    ("temperature_2m_mean", "float", "Monthly mean 2m temperature"),
    ("temperature_2m_max", "float", "Monthly maximum 2m temperature"),
    ("temperature_2m_min", "float", "Monthly minimum 2m temperature"),
    ("precipitation_sum", "float", "Monthly total precipitation"),
    ("rain_sum", "float", "Monthly total rain"),
    ("precipitation_hours", "float", "Monthly hours with precipitation"),
    ("retail_index", "float", "Retail sales volume index, 2021=100"),
    ("unemployment_rate", "float", "Seasonally adjusted unemployment rate"),
    ("inflation_yoy", "float", "HICP year-over-year inflation"),
    ("CCI", "float", "OECD amplitude-adjusted Consumer Confidence Index"),
    ("covid_dummy", "int", "1 for 2020-03 to 2022-06, otherwise 0"),
    ("month_num", "int", "Calendar month number"),
    ("month_sin", "float", "Cyclical month encoding, sine component"),
    ("month_cos", "float", "Cyclical month encoding, cosine component"),
]


def season_name(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def find_data_file(name):
    """Find a data file in data/ first, then in ROOT (backward compatibility)."""
    candidates = [DATA_DIR / name, ROOT / name]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not locate '{name}'. Tried: {[str(c) for c in candidates]}"
    )


def load_panel():
    panel = pd.read_csv(find_data_file("panel.csv"))
    panel["year_month"] = pd.to_datetime(panel["year_month"])
    panel["year_month_label"] = panel["year_month"].dt.strftime("%Y-%m")
    panel["season"] = panel["year_month"].dt.month.map(season_name)
    panel["year"] = panel["year_month"].dt.year
    return panel


def load_model_results():
    results = pd.read_csv(find_data_file("model_comparison.csv"))
    return results.rename(columns={"R²": "R2"})


PANEL = load_panel()
MODEL_RESULTS = load_model_results()


def plot_html(fig, height=None):
    if height:
        fig.update_layout(height=height)
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=42, r=22, t=48, b=42),
        font=dict(family="Inter, Arial, sans-serif"),
        legend_title_text="",
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": False, "responsive": True},
    )


def stat_cards(items):
    return items


def table_html(df, classes="data-table"):
    return df.to_html(index=False, classes=classes, border=0, escape=False)


def figure_exists(path):
    """Check whether the requested figure exists relative to ROOT or figures/."""
    candidates = [ROOT / path, FIGURES_DIR / Path(path).name]
    return any(c.exists() for c in candidates)


def figure_html(path, caption):
    """Render an <img> only if the file exists, otherwise return an empty string.
    This prevents broken image icons for missing notebook renders.
    """
    if not figure_exists(path):
        return ""
    src = f"/project-figures/{path}"
    return (
        f'<figure class="inline-figure"><img src="{src}" alt="{caption}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def image_gallery_items(images):
    """Filter out missing images so the gallery is never broken."""
    return [(p, cap) for p, cap in images if figure_exists(p)]


def hero(page_title, subtitle, icon, compact=True):
    return {
        "title": page_title,
        "subtitle": subtitle,
        "icon": icon,
        "compact": compact,
    }


def render_page(active, title, hero_data, sections):
    return render_template(
        "page.html",
        nav_items=NAV_ITEMS,
        active=active,
        title=title,
        hero=hero_data,
        sections=sections,
        repo_url="https://github.com/erkamaygun-stack/dsa210-34328-erkam-sait-project",
    )


@app.route("/project-figures/<path:filename>")
def project_figures(filename):
    """Resolve image paths in this order:
    1. ROOT/<filename>            (full relative paths like 'figures/ml_02_...')
    2. ROOT/figures/<basename>    (bare names like 'ml_06_...')
    """
    # Try the exact path first
    target = ROOT / filename
    if target.exists() and target.is_file():
        return send_from_directory(target.parent, target.name)
    # Fall back to figures/ directory
    fallback = FIGURES_DIR / Path(filename).name
    if fallback.exists() and fallback.is_file():
        return send_from_directory(FIGURES_DIR, fallback.name)
    abort(404)


@app.route("/_stcore/health")
def health():
    return "ok"


@app.route("/_stcore/host-config")
def host_config():
    return jsonify({})


@app.route("/_stcore/stream")
def stream_compat():
    return ("", 204)


@app.route("/")
def home():
    cci_fig = px.line(
        PANEL,
        x="year_month",
        y="CCI",
        color="country",
        color_discrete_map=COUNTRY_COLORS,
        title="Consumer Confidence Index by Country",
    )
    sections = [
        {
            "type": "home_overview",
            "stats": stat_cards(
                [
                    ("10", "Years", "Monthly data from 2015 to 2024", "fa-calendar"),
                    ("480", "Panel Rows", "Four countries x 120 months", "fa-table"),
                    ("8", "Weather Variables", "Sunshine, daylight, temperature and rain", "fa-cloud-sun"),
                    ("0.871", "Best Retail R2", "Random Forest cross-validation", "fa-trophy"),
                ]
            ),
            "chart": plot_html(cci_fig, 430),
        },
        {
            "type": "research_cards",
            "title": "Explore the Research",
            "cards": [
                ("Motivation", "Why weather, confidence and retail behavior belong in the same research question.", "motivation", "fa-compass"),
                ("Data Sources", "Open-Meteo, OECD and Eurostat merged into one country-month panel.", "data_sources", "fa-database"),
                ("Methodology", "Cleaning decisions, missing values, feature engineering and modeling pipeline.", "methodology", "fa-gears"),
                ("EDA", "Target distributions, correlation heatmap, boxplots and time-series figures.", "eda", "fa-chart-area"),
                ("Findings", "Hypothesis tests with p-values and plain-English decisions.", "findings", "fa-flask"),
                ("ML Results", "Model comparison, feature importance and actual-vs-predicted diagnostics.", "ml_results", "fa-robot"),
                ("Explorer", "Interactive variable explorer for the final panel.", "explorer", "fa-sliders"),
                ("Conclusions", "Limitations, future work and the final project interpretation.", "conclusions", "fa-flag-checkered"),
            ],
        },
    ]
    return render_page(
        "home",
        "Sunshine, Sentiment & Spending",
        {
            "home": True,
            "title": "Sunshine, Sentiment & Spending",
            "subtitle": "A DSA210 data science project testing whether weather and macroeconomic context can move consumer confidence and retail behavior across Germany, Spain, Turkey and the United Kingdom (2015–2024).",
        },
        sections,
    )


@app.route("/motivation")
def motivation():
    sections = [
        {
            "type": "text_card",
            "title": "Research Motivation",
            "icon": "fa-compass",
            "body": """
            This project started from a personal curiosity. I study data science with a strong interest
            in psychology and decision economics — the way mood, environment and macroeconomic pressure
            quietly shape the everyday choices people make. I wanted to move past intuition and look at
            whether those forces actually leave a measurable trace in real data. So I built a country-level
            monthly panel covering Germany, Spain, Turkey and the UK, and asked the question quantitatively:
            do sunshine, temperature and macro context shift consumer confidence and retail behavior in
            ways I can actually detect?
            """,
        },
        {
            "type": "three_cards",
            "title": "Research Questions",
            "cards": [
                (
                    "Does same-month sunshine correlate with consumer confidence?",
                    "Folk wisdom says sunny days lift mood. Part A tests this directly with Pearson and Spearman correlations between monthly sunshine hours and CCI for each of the four countries.",
                    "fa-sun",
                ),
                (
                    "Does last month's weather predict this month's mood?",
                    "If weather effects need time to surface in survey responses, a lagged relationship should appear. Lag-1 correlation tests whether previous-month sunshine moves current-month CCI.",
                    "fa-clock-rotate-left",
                ),
                (
                    "Do summer and winter consumer confidence systematically differ?",
                    "The most extreme weather contrast should reveal the strongest effect. A two-sample Welch t-test compares summer (Jun–Aug) and winter (Dec–Feb) CCI within each country.",
                    "fa-cloud-sun",
                ),
                (
                    "Can non-linear ML models recover signals that linear methods miss?",
                    "Pearson catches only straight-line patterns. Random Forest, Gradient Boosting, Decision Tree and KNN test whether weather plus macro context predicts CCI and retail when the relationship is non-linear or interactive.",
                    "fa-diagram-project",
                ),
                (
                    "Is consumer perception or consumer behavior more responsive to weather?",
                    "CCI (a smoothed survey index) and retail volume (real spending) are two different windows into the same population. Comparing prediction performance shows which one weather and macro variables actually move.",
                    "fa-scale-balanced",
                ),
                (
                    "Can countries be grouped by climate-driven consumer profiles?",
                    "K-Means and hierarchical clustering test whether the 480 country-months separate into meaningful weather-and-behavior archetypes — and whether those archetypes match real economic groupings.",
                    "fa-layer-group",
                ),
            ],
        },
        {
            "type": "text_card",
            "title": "Why Data Science?",
            "icon": "fa-chart-line",
            "body": """
            I wanted this project to do more than just visualise the question. Answering it properly meant
            integrating four public data sources, making cleaning decisions transparent, running formal
            hypothesis tests, building predictive models and being honest about what the numbers cannot tell
            me. That full pipeline — not any single chart — is what convinced me the answer is more nuanced
            than the intuition I started with.
            """,
        },
    ]
    return render_page("motivation", "Motivation", hero("Motivation", "Why this weather-and-consumer project matters", "fa-compass"), sections)


@app.route("/data-sources")
def data_sources():
    dictionary = pd.DataFrame(DATA_DICTIONARY, columns=["Column", "Type", "Meaning"])
    coverage = PANEL.groupby("country").agg(
        Rows=("country", "size"),
        First_Month=("year_month_label", "min"),
        Last_Month=("year_month_label", "max"),
        Missing_Retail=("retail_index", lambda s: int(s.isna().sum())),
        Missing_CCI=("CCI", lambda s: int(s.isna().sum())),
    ).reset_index()
    sections = [
        {
            "type": "source_grid",
            "sources": [
                ("Open-Meteo Historical Weather API", "Population-weighted weather from five cities per country.", "8 variables", "fa-cloud-sun"),
                ("OECD Composite Leading Indicators", "Amplitude-adjusted Consumer Confidence Index.", "CCI target", "fa-building-columns"),
                ("Eurostat Retail Sales", "Seasonally and working-day adjusted retail volume index, 2021=100.", "Retail target", "fa-store"),
                ("Eurostat Macro Controls", "Unemployment, inflation and derived COVID period indicator.", "Controls", "fa-chart-simple"),
            ],
        },
        {"type": "pipeline"},
        {"type": "table", "title": "Data Dictionary", "html": table_html(dictionary)},
        {"type": "table", "title": "Coverage by Country", "html": table_html(coverage)},
    ]
    return render_page("data_sources", "Data Sources", hero("Data Sources", "Four integrated sources, one reproducible country-month panel", "fa-database"), sections)


@app.route("/methodology")
def methodology():
    missing = PANEL.isna().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    missing["Missing Percent"] = (missing["Missing Values"] / len(PANEL) * 100).round(2)
    missing = missing[missing["Missing Values"] > 0]

    outliers = []
    for col in PANEL.select_dtypes(include=np.number).columns:
        q1, q3 = PANEL[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers.append((col, int(((PANEL[col] < low) | (PANEL[col] > high)).sum())))
    outlier_df = pd.DataFrame(outliers, columns=["Column", "IQR Outliers"]).sort_values("IQR Outliers", ascending=False)

    sections = [
        {
            "type": "timeline",
            "title": "Data Processing Pipeline",
            "items": [
                ("Collect", "Fetch weather, CCI, retail and macroeconomic data from public sources."),
                ("Merge", "Join sources by country and year_month into a 480-row panel."),
                ("Clean", "Keep target-specific missingness explicit instead of hiding source gaps."),
                ("Engineer", "Add COVID dummy, cyclical month features, inflation log and weather PCA in notebooks."),
                ("Evaluate", "Use hypothesis tests, regression, tree ensembles, classification and clustering."),
            ],
        },
        {"type": "table", "title": "Missing Value Diagnostics", "html": table_html(missing)},
        {
            "type": "insight",
            "tone": "warning",
            "title": "Cleaning Decision",
            "body": "Retail has 120 missing rows because UK retail observations are unavailable in the merged target. Retail models use available Germany, Spain and Turkey observations; CCI analysis keeps all four countries.",
        },
        {"type": "table", "title": "Outlier Diagnostics", "html": table_html(outlier_df)},
        {
            "type": "feature_cards",
            "title": "Feature Engineering",
            "cards": [
                ("covid_dummy", "Encodes March 2020 to June 2022 as a structural shock period.", "fa-virus-covid"),
                ("month_sin / month_cos", "Captures seasonality while keeping December and January close.", "fa-repeat"),
                ("inflation_log", "Reduces high-inflation leverage while preserving information.", "fa-money-bill-trend-up"),
                ("weather_PC1 / weather_PC2", "Compresses correlated weather variables into orthogonal components.", "fa-layer-group"),
            ],
        },
        {
            "type": "image_gallery",
            "title": "Feature Engineering and Dimensionality Reduction Figures",
            "images": image_gallery_items([
                ("figures/ml_02_pca_weather.png", "Weather PCA scree plot and PC1/PC2 loadings"),
                ("figures/ml_01_correlation_heatmap.png", "Full correlation heatmap before PCA"),
            ]),
        },
    ]
    return render_page("methodology", "Methodology", hero("Methodology", "From raw public data to ML-ready evidence", "fa-gears"), sections)


@app.route("/eda")
def eda():
    cci_hist = px.histogram(PANEL, x="CCI", color="country", marginal="box", nbins=38, color_discrete_map=COUNTRY_COLORS, title="Target Distribution: CCI")
    retail_hist = px.histogram(PANEL.dropna(subset=["retail_index"]), x="retail_index", color="country", marginal="box", nbins=38, color_discrete_map=COUNTRY_COLORS, title="Target Distribution: Retail Index")
    corr_cols = WEATHER_FEATURES + ["unemployment_rate", "inflation_yoy", "CCI", "retail_index"]
    heatmap = px.imshow(PANEL[corr_cols].corr(numeric_only=True), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Heatmap")
    box = px.box(PANEL.dropna(subset=["retail_index"]), x="season", y="retail_index", color="country", color_discrete_map=COUNTRY_COLORS, title="Season vs Retail Index")
    weather_line = px.line(PANEL, x="year_month", y="sunshine_hours", color="country", color_discrete_map=COUNTRY_COLORS, title="Monthly Sunshine Hours")
    sections = [
        {
            "type": "figure_grid",
            "title": "Target Variables",
            "figures": [
                ("Figure 1: CCI Distribution", plot_html(cci_hist, 430), "CCI is tightly centered around 100 because the OECD series is amplitude-adjusted and smoothed."),
                ("Figure 2: Retail Distribution", plot_html(retail_hist, 430), "Retail index varies more strongly by country and period, making it easier to predict than CCI."),
            ],
        },
        {
            "type": "single_figure",
            "title": "Figure 3: Correlation Matrix",
            "chart": plot_html(heatmap, 690),
            "insight": "Weather variables are strongly correlated with each other, which justifies PCA before linear modeling.",
        },
        {
            "type": "figure_grid",
            "title": "Categorical and Time-Series Views",
            "figures": [
                ("Figure 4: Categorical vs Target", plot_html(box, 430), "Seasonal differences are visible, but CCI seasonal tests remain statistically non-significant."),
                ("Figure 5: Weather Time Series", plot_html(weather_line, 430), "Sunshine has a strong seasonal cycle, while the consumer outcomes respond less directly."),
            ],
        },
        {
            "type": "image_gallery",
            "title": "Notebook EDA Figures",
            "images": image_gallery_items([
                ("figures/ml_01_correlation_heatmap.png", "ML Figure 1: full correlation heatmap"),
                ("figures/ml_02_pca_weather.png", "ML Figure 2: weather PCA and loadings"),
            ]),
        },
    ]
    return render_page("eda", "Exploratory Data Analysis", hero("Exploratory Data Analysis", "Visual evidence before statistical testing and modeling", "fa-chart-area"), sections)


def hypothesis_results():
    rows = []
    for country, grp in PANEL.groupby("country"):
        clean = grp.dropna(subset=["sunshine_hours", "CCI"])
        if len(clean) < 3:
            rows.append([country, np.nan, np.nan, np.nan, np.nan, "Insufficient data"])
            continue
        pearson_r, pearson_p = stats.pearsonr(clean["sunshine_hours"], clean["CCI"])
        spearman_r, spearman_p = stats.spearmanr(clean["sunshine_hours"], clean["CCI"])
        rows.append([country, pearson_r, pearson_p, spearman_r, spearman_p, "Rejected" if pearson_p < 0.05 else "Not rejected"])
    corr = pd.DataFrame(rows, columns=["Country", "Pearson r", "Pearson p", "Spearman r", "Spearman p", "H0"])

    lag_rows = []
    for country, grp in PANEL.sort_values("year_month").groupby("country"):
        lagged = grp.copy()
        lagged["sunshine_lag1"] = lagged["sunshine_hours"].shift(1)
        clean = lagged.dropna(subset=["sunshine_lag1", "CCI"])
        if len(clean) < 3:
            lag_rows.append([country, np.nan, np.nan, "Insufficient data"])
            continue
        r, p = stats.pearsonr(clean["sunshine_lag1"], clean["CCI"])
        lag_rows.append([country, r, p, "Rejected" if p < 0.05 else "Not rejected"])
    lag = pd.DataFrame(lag_rows, columns=["Country", "Lag-1 r", "p-value", "H0"])

    t_rows = []
    for country, grp in PANEL.groupby("country"):
        summer = grp[grp["season"] == "Summer"]["CCI"].dropna()
        winter = grp[grp["season"] == "Winter"]["CCI"].dropna()
        if len(summer) < 2 or len(winter) < 2:
            t_rows.append([country, np.nan, np.nan, np.nan, np.nan, "Insufficient data"])
            continue
        t_stat, p_value = stats.ttest_ind(summer, winter, equal_var=False)
        t_rows.append([country, summer.mean(), winter.mean(), t_stat, p_value, "Rejected" if p_value < 0.05 else "Not rejected"])
    seasonal = pd.DataFrame(t_rows, columns=["Country", "Summer CCI", "Winter CCI", "t-stat", "p-value", "H0"])
    return corr.round(3), lag.round(3), seasonal.round(3)


@app.route("/findings")
def findings():
    corr, lag, seasonal = hypothesis_results()
    variability = PANEL.groupby("country").agg(sunshine_std=("sunshine_hours", "std"), cci_std=("CCI", "std")).reset_index()
    var_fig = px.scatter(
        variability,
        x="sunshine_std",
        y="cci_std",
        text="country",
        color="country",
        color_discrete_map=COUNTRY_COLORS,
        title="Cross-country Variability",
    )
    var_fig.update_traces(textposition="top center")
    sections = [
        {
            "type": "hypothesis_cards",
            "items": [
                (
                    "H1",
                    "Same-month sunshine is associated with CCI",
                    "Not supported",
                    "All country-level Pearson p-values are greater than 0.05.",
                    "weak",
                    table_html(corr),
                ),
                (
                    "H2",
                    "Previous-month sunshine predicts current-month CCI",
                    "Not supported",
                    "Lag-1 sunshine also fails to reject H0.",
                    "weak",
                    table_html(lag),
                ),
                (
                    "H3",
                    "Summer and winter CCI means differ",
                    "Not supported",
                    "Seasonal t-tests are non-significant for all countries.",
                    "weak",
                    table_html(seasonal),
                ),
                (
                    "H4",
                    "Countries with more volatile sunshine have different CCI volatility",
                    "Supported, cautiously",
                    "The four-country variability check is descriptive due to small sample (n=4).",
                    "confirmed",
                    plot_html(var_fig, 410),
                ),
            ],
        },
        {
            "type": "insight",
            "tone": "success",
            "title": "Scientific Interpretation",
            "body": "The absence of simple linear significance does not mean weather is irrelevant. It means the relationship is likely indirect, non-linear and confounded with macroeconomic context and seasonality.",
        },
    ]
    return render_page("findings", "Research Findings", hero("Research Findings", "Hypothesis tests, p-values and defensible interpretation", "fa-flask"), sections)


def actual_predicted():
    df = PANEL.dropna(subset=WEATHER_FEATURES + ["unemployment_rate", "inflation_yoy", "retail_index"]).copy()
    df["inflation_log"] = np.log1p(df["inflation_yoy"].clip(lower=0))
    features = WEATHER_FEATURES + ["unemployment_rate", "inflation_log", "covid_dummy", "month_sin", "month_cos"]

    X = df[features]
    y = df["retail_index"]
    country_series = df["country"]

    X_train, X_test, y_train, y_test, _, country_test = train_test_split(
        X, y, country_series, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    out = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": pred,
        "country": country_test.values,
    })
    metrics = {
        "R2": r2_score(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE": mean_absolute_error(y_test, pred),
    }
    return out, metrics


@app.route("/ml-results")
def ml_results():
    results = MODEL_RESULTS.copy()
    results["RMSE"] = results["RMSE"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    results["R2 / AUC"] = results["R2"].map(lambda x: f"{x:.3f}")
    results = results.drop(columns=["R2"])

    pred_df, metrics = actual_predicted()
    pred_fig = px.scatter(
        pred_df,
        x="Actual",
        y="Predicted",
        color="country",
        color_discrete_map=COUNTRY_COLORS,
        title="Actual vs Predicted Retail Index",
    )
    min_v = float(min(pred_df["Actual"].min(), pred_df["Predicted"].min()))
    max_v = float(max(pred_df["Actual"].max(), pred_df["Predicted"].max()))
    pred_fig.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            name="Perfect prediction",
            line=dict(color="#111827", dash="dash"),
        )
    )

    sections = [
        {"type": "table", "title": "Model Performance Comparison", "html": table_html(results)},
        {
            "type": "ml_metrics",
            "metrics": [
                ("Random Forest Retail", "0.871", "Best CV R2 in notebook", "fa-trophy"),
                ("Random Forest CCI", "0.324", "Moderate non-linear CCI signal", "fa-tree"),
                ("Logistic CCI Direction", "0.469", "AUC below random", "fa-triangle-exclamation"),
                ("Holdout Retail RMSE", f"{metrics['RMSE']:.2f}", f"R2={metrics['R2']:.3f}, MAE={metrics['MAE']:.2f}", "fa-bullseye"),
            ],
        },
        {
            "type": "image_gallery",
            "title": "Regression and Classification Diagnostics",
            "images": image_gallery_items([
                ("figures/ml_03_lr_coefficients_cci.png", "ML Figure 3: linear regression coefficients for CCI"),
                ("figures/ml_04_logreg_confusion.png", "ML Figure 4: logistic regression confusion matrix"),
                ("figures/ml_11_roc_curve.png", "ML Figure 11: ROC curve diagnostic"),
            ]),
        },
        {
            "type": "image_gallery",
            "title": "Feature Importance Figures",
            "images": image_gallery_items([
                ("figures/ml_05_rf_importance_cci.png", "Random Forest feature importance for CCI"),
                ("figures/ml_06_rf_importance_retail.png", "Random Forest feature importance for retail"),
            ]),
        },
        {
            "type": "image_gallery",
            "title": "Model Comparison and Additional Model Figures",
            "images": image_gallery_items([
                ("figures/ml_10_model_comparison.png", "ML Figure 10: model comparison, RMSE and R2"),
                ("figures/ml_15_model_comparison.png", "ML Figure 15: extended model comparison"),
                ("figures/ml_12_decision_tree.png", "Decision tree visualization"),
                ("figures/ml_13_knn_k_selection.png", "KNN k-selection"),
            ]),
        },
        {
            "type": "image_gallery",
            "title": "Clustering and Country Climate Profile Figures",
            "images": image_gallery_items([
                ("figures/ml_07_kmeans_choice.png", "ML Figure 7: K-Means elbow and silhouette choice"),
                ("figures/ml_08_cluster_distribution.png", "ML Figure 8: cluster distribution by country"),
                ("figures/ml_09_hierarchical_dendrogram.png", "ML Figure 9: hierarchical dendrogram of country climate profiles"),
            ]),
        },
        {
            "type": "single_figure",
            "title": "Actual vs Predicted Diagnostic",
            "chart": plot_html(pred_fig, 520),
            "insight": "The retail model sits much closer to the diagonal than linear baselines, but this should be read as predictive association, not causal weather effect.",
        },
        {
            "type": "insight",
            "tone": "warning",
            "title": "Why Weak Models Matter",
            "body": "Linear regression and CCI direction classification are intentionally shown. Negative or weak results demonstrate that the project is testing claims rather than only displaying successful models.",
        },
    ]
    return render_page("ml_results", "Machine Learning Results", hero("Machine Learning Results", "Model comparison, feature importance and honest limitations", "fa-robot"), sections)


@app.route("/explorer")
def explorer():
    x_var = request.args.get("x", "sunshine_hours")
    y_var = request.args.get("y", "CCI")
    selected = request.args.getlist("country") or COUNTRIES
    allowed_x = WEATHER_FEATURES + CONTROL_FEATURES
    allowed_y = ["CCI", "retail_index", "sunshine_hours", "temperature_2m_mean", "inflation_yoy"]
    x_var = x_var if x_var in allowed_x else "sunshine_hours"
    y_var = y_var if y_var in allowed_y else "CCI"
    chart_df = PANEL[PANEL["country"].isin(selected)].dropna(subset=[x_var, y_var])
    fig = px.scatter(
        chart_df,
        x=x_var,
        y=y_var,
        color="country",
        hover_data=["year_month_label", "season"],
        color_discrete_map=COUNTRY_COLORS,
        title=f"{y_var} vs {x_var}",
    )
    sections = [
        {
            "type": "explorer",
            "chart": plot_html(fig, 560),
            "x_var": x_var,
            "y_var": y_var,
            "selected": selected,
            "allowed_x": allowed_x,
            "allowed_y": allowed_y,
            "countries": COUNTRIES,
        },
        {
            "type": "insight",
            "tone": "success",
            "title": "How to Read It",
            "body": "Use the controls to test whether a relationship is stable across targets and countries. The strongest patterns usually appear when macro controls are part of the story.",
        },
    ]
    return render_page("explorer", "Interactive Explorer", hero("Interactive Explorer", "Choose variables and inspect the panel yourself", "fa-sliders"), sections)


@app.route("/conclusions")
def conclusions():
    sections = [
        {
            "type": "conclusion_cards",
            "items": [
                ("Raw weather-CCI evidence is weak", "Pearson, Spearman, lag and seasonal t-tests do not support a direct linear CCI story.", "fa-circle-xmark"),
                ("Retail is more predictable", "Random Forest reaches strong retail performance once weather and macro controls are combined.", "fa-store"),
                ("Macro context dominates", "Inflation and unemployment carry major signal; weather contributes but is not the sole driver.", "fa-chart-line"),
                ("Claims remain associative", "The project does not claim causality; it builds a transparent predictive and statistical analysis.", "fa-scale-balanced"),
            ],
        },
        {
            "type": "three_cards",
            "title": "Future Work",
            "cards": [
                ("Time-series validation", "Use rolling-origin or blocked time splits instead of shuffled K-Fold.", "fa-clock"),
                ("Sub-national data", "Add city or regional retail proxies to reduce country aggregation bias.", "fa-map-location-dot"),
                ("Causal design", "Use event-study or instrumental-variable logic before making causal claims.", "fa-microscope"),
            ],
        },
        {
            "type": "final_cta",
            "title": "Thank you for exploring the project",
            "body": "Weather may not move confidence in a simple straight line, but the full pipeline shows how data science can separate weak direct evidence from stronger predictive structure.",
        },
    ]
    return render_page("conclusions", "Conclusions", hero("Conclusions", "Key findings, limitations and next steps", "fa-flag-checkered"), sections)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=False)
