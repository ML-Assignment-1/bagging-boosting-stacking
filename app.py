"""
Sri Lanka House Price Predictor: an ensemble learning dashboard.

Front end for the models trained in ML_assignment.ipynb.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent

FAMILY_COLOURS = {
    "Baseline": "#898781",
    "Bagging": "#2a78d6",
    "Boosting": "#eb6834",
    "Hybrid": "#eda100",
    "Stacking": "#e87ba4",
}

MODEL_FAMILY = {
    "Random Forest": "Bagging",
    "XGBoost": "Boosting",
    "CatBoost": "Boosting",
    "Hybrid RF + CatBoost": "Hybrid",
    "Stacking": "Stacking",
}

PRICE_COL = "Price (LKR M)"
BASELINE = "Median Baseline"

st.set_page_config(
    page_title="Sri Lanka House Price Predictor",
    page_icon="🏠",
    layout="wide",
)


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_model():
    return joblib.load(BASE / "models" / "stacking_model.pkl")


@st.cache_data
def load_metadata():
    with open(BASE / "models" / "metadata.json") as f:
        return json.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "data" / "clean_houses.csv", parse_dates=["posted_date"])
    df["price_million"] = df["price_lkr"] / 1_000_000
    return df


stack_model = load_model()
meta = load_metadata()
houses = load_data()
district_towns = meta["district_towns"]
metrics = pd.DataFrame(meta["metrics"])


def predict_all(row: pd.DataFrame) -> pd.DataFrame:
    """Price from every model, in LKR millions."""
    base = {name: float(est.predict(row)[0]) for name, est in stack_model.named_estimators_.items()}
    w = meta["blend_weight_rf"]
    out = {
        "Random Forest": base["Random Forest"],
        "XGBoost": base["XGBoost"],
        "CatBoost": base["CatBoost"],
        "Hybrid RF + CatBoost": w * base["Random Forest"] + (1 - w) * base["CatBoost"],
        "Stacking": float(stack_model.predict(row)[0]),
    }
    df = pd.DataFrame({"Model": list(out), PRICE_COL: list(out.values())})
    df["Family"] = df["Model"].map(MODEL_FAMILY)
    return df


# ------------------------------------------------------------------ header
st.title("🏠 Sri Lanka House Price Predictor")
st.caption(
    f"Ensemble models (bagging, boosting, a hybrid and stacking) trained on {meta['n_train']:,} "
    f"ikman.lk house ads ({meta['train_period'][0]} to {meta['train_period'][1]}) and tested on "
    f"the newest {meta['n_test']:,} ads."
)

tab_predict, tab_compare, tab_market, tab_about = st.tabs(
    ["Predict a price", "Compare models", "Market insights", "About"]
)

# ------------------------------------------------------------------ predict
with tab_predict:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Describe the house")
        districts = sorted(district_towns)
        district = st.selectbox(
            "District", districts,
            index=districts.index("Colombo") if "Colombo" in districts else 0,
        )
        towns = district_towns[district]
        town = st.selectbox("Town", towns)
        c1, c2 = st.columns(2)
        bedrooms = c1.number_input("Bedrooms", 1, 10, 3)
        bathrooms = c2.number_input("Bathrooms", 1, 10, 2)
        house_sqft = st.number_input("House size (sqft)", 300, 20_000, 2_000, step=100)
        land_perches = st.number_input("Land size (perches)", 1.0, 500.0, 10.0, step=0.5)

    row = pd.DataFrame([{
        "district": district,
        "location": town,
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "house_sqft": float(house_sqft),
        "land_perches": float(land_perches),
    }])
    preds = predict_all(row)
    final = preds.loc[preds["Model"] == "Stacking", PRICE_COL].iloc[0]
    stack_mae = metrics.loc[metrics["Model"] == "Stacking", "Test MAE"].iloc[0]
    town_prices = houses.loc[houses["location"] == town, "price_million"]

    with right:
        st.subheader("Estimated asking price")
        m1, m2, m3 = st.columns(3)
        m1.metric("Stacked ensemble", f"Rs {final:,.1f} M")
        m2.metric(
            "Average error (test MAE)",
            f"± Rs {stack_mae:,.1f} M",
            help="Average absolute error of the stacked model on the newest (unseen) ads.",
        )
        if len(town_prices) >= 5:
            m3.metric(
                f"Median ad price in {town}",
                f"Rs {town_prices.median():,.1f} M",
                help=f"Based on {len(town_prices):,} listings.",
            )

        fig = px.bar(
            preds,
            x=PRICE_COL,
            y="Model",
            color="Family",
            orientation="h",
            color_discrete_map=FAMILY_COLOURS,
            text=preds[PRICE_COL].map(lambda v: f"Rs {v:,.1f} M"),
            category_orders={"Model": list(preds["Model"])},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            title="What each model thinks",
            height=340,
            margin={"l": 10, "r": 60, "t": 50, "b": 10},
            legend_title_text="",
            xaxis_title="Price (LKR millions)",
            yaxis_title="",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "The stacked ensemble combines Random Forest, XGBoost and CatBoost, using weights "
            "learned by a Ridge meta-model. When the models disagree a lot, treat the estimate with care."
        )

# ------------------------------------------------------------------ compare
with tab_compare:
    st.subheader("Results and overfitting check")

    models_only = metrics[metrics["Model"] != BASELINE]
    best = models_only.iloc[0]
    baseline = metrics[metrics["Model"] == BASELINE].iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Best model (test R²)", best["Model"])
    k2.metric("Best test R²", f"{best['Test R2']:.3f}")
    k3.metric("Its test MAE", f"Rs {best['Test MAE']:.2f} M",
              f"{best['Test MAE'] - baseline['Test MAE']:.2f} M vs baseline", delta_color="inverse")

    c1, c2 = st.columns(2)
    fit_plot = models_only.melt(
        id_vars=["Model"], value_vars=["Train R2", "Test R2"], var_name="Set", value_name="R²"
    )
    fit_plot["Set"] = fit_plot["Set"].str.replace(" R2", "")
    fig = px.bar(
        fit_plot, x="R²", y="Model", color="Set", barmode="group", orientation="h",
        color_discrete_map={"Train": "#2a78d6", "Test": "#eb6834"},
        text=fit_plot["R²"].map(lambda v: f"{v:.3f}"),
        category_orders={"Model": list(models_only["Model"])},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(title="Train vs test R² (bigger gap = more overfitting)", height=420,
                      legend_title_text="", yaxis_title="", margin={"l": 10, "r": 50, "t": 50, "b": 10})
    fig.update_xaxes(range=[0.5, 1.02])
    c1.plotly_chart(fig, width="stretch")

    fig = px.bar(
        models_only, x="Test MAE", y="Model", color="Family", orientation="h",
        color_discrete_map=FAMILY_COLOURS,
        text=models_only["Test MAE"].map(lambda v: f"{v:.2f}"),
        category_orders={"Model": list(models_only["Model"])},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(title="Test MAE in LKR M (lower is better)", height=420, legend_title_text="",
                      yaxis_title="", margin={"l": 10, "r": 50, "t": 50, "b": 10})
    c2.plotly_chart(fig, width="stretch")

    st.dataframe(
        metrics.drop(columns=["Family"]).rename(columns={
            "Train R2": "Train R²", "Test R2": "Test R²", "Gap": "Gap (R²)",
            "Train MAE": "Train MAE (LKR M)", "Test MAE": "Test MAE (LKR M)", "Test RMSE": "Test RMSE (LKR M)",
        }).round(3),
        hide_index=True, width="stretch",
    )
    st.caption(
        "Gap = train R² − test R². A big gap means the model memorised the training ads (overfitting); "
        "low scores on both mean it is too simple (underfitting). Part of the gap also comes from prices "
        "changing over time, because the test ads are the newest ones."
    )

    st.markdown("**Stacking: weight given to each base model**")
    weights = pd.DataFrame({"Model": list(meta["stack_weights"]), "Weight": list(meta["stack_weights"].values())})
    weights["Family"] = weights["Model"].map(MODEL_FAMILY)
    fig = px.bar(weights, x="Weight", y="Model", color="Family", orientation="h",
                 color_discrete_map=FAMILY_COLOURS, text=weights["Weight"].round(2))
    fig.update_layout(height=240, legend_title_text="", yaxis_title="", margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Why the models behave this way")
    st.markdown(
        """
- **Bagging (Random Forest) reduces variance.** Averaging 60 trees trained on bootstrap samples smooths out single-tree mistakes, but it doesn't reduce bias, so RF is the weakest ensemble here (test R² 0.730).
- **Boosting reduces bias, so it beats bagging.** Each new tree fixes what the earlier ones got wrong, which captures interactions like *town × size*. CatBoost (0.746) edges out XGBoost (0.742).
- **The hybrid (0.35 × RF + 0.65 × CatBoost) is the best model** (test R² 0.750, MAE Rs 10.4M). It beats both of its parts, because the smoother Random Forest cancels some of CatBoost's overshoots.
- **Stacking (0.749) practically ties the hybrid.** Its meta-model gives CatBoost most of the weight (0.78). The gain is small because all three base models are tree ensembles with correlated errors.
- **No model clearly overfits.** Every ensemble shows only a slight train-test gap (0.07–0.08), kept small by depth limits, subsampling and small learning rates. Even the training R² tops out around 0.83, because similar houses are advertised at quite different prices.
"""
    )

# ------------------------------------------------------------------ market
with tab_market:
    st.subheader("What the ikman.lk data shows")
    sel = st.multiselect("Filter districts (empty = all)", sorted(houses["district"].unique()))
    view = houses[houses["district"].isin(sel)] if sel else houses

    c1, c2 = st.columns(2)
    med = (view.groupby("district")["price_million"].agg(["median", "size"])
           .query("size >= 20").sort_values("median", ascending=False).reset_index())
    fig = px.bar(med, x="district", y="median", color_discrete_sequence=["#2a78d6"],
                 labels={"median": "Median price (LKR M)", "district": ""},
                 hover_data={"size": True})
    fig.update_layout(title="Median asking price by district", height=420, margin={"t": 50, "b": 10})
    c1.plotly_chart(fig, width="stretch")

    fig = px.histogram(view[view["price_million"] <= 200], x="price_million", nbins=60,
                       color_discrete_sequence=["#2a78d6"],
                       labels={"price_million": "Asking price (LKR M)"})
    fig.update_layout(title="Price distribution (up to Rs 200M)", height=420, yaxis_title="Listings",
                      margin={"t": 50, "b": 10}, bargap=0.05)
    c2.plotly_chart(fig, width="stretch")

    sample = view.sample(min(3000, len(view)), random_state=42)
    fig = px.scatter(sample, x="house_sqft", y="price_million", opacity=0.35,
                     color_discrete_sequence=["#2a78d6"],
                     labels={"house_sqft": "House size (sqft)", "price_million": "Price (LKR M)"},
                     hover_data=["district", "location", "bedrooms"])
    fig.update_layout(title="House size vs price (sample)", height=420, margin={"t": 50, "b": 10})
    fig.update_yaxes(range=[0, 250])
    fig.update_xaxes(range=[0, 8000])
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(
        f"""
### About this project
**Data:** house-for-sale ads scraped from [ikman.lk](https://ikman.lk), from a public Kaggle dataset. After cleaning
(price parsing, outlier removal, dropping reposted ads) there are {meta['n_train'] + meta['n_test']:,} listings.

**Target:** asking price in LKR millions. Note that this is the *asking* price, not the final sale price.

**Split:** by date. The models are trained on ads from {meta['train_period'][0]} to {meta['train_period'][1]} and
tested on ads from {meta['test_period'][0]} to {meta['test_period'][1]}.

**Models:**
- Baseline: median training price
- Bagging: Random Forest
- Boosting: XGBoost, CatBoost
- Hybrid: weighted Random Forest + CatBoost blend (weight {meta['blend_weight_rf']:.2f} on RF)
- Stacking: Random Forest + XGBoost + CatBoost, combined by a Ridge meta-model
"""
    )

st.divider()
st.caption("Built with Streamlit · Models trained in ML_assignment.ipynb")
