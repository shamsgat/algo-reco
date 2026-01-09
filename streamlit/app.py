import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(
    page_title="Dashboard Business – Substitution",
    layout="wide"
)

# ----------------------
# LOAD DATA
# ----------------------
@st.cache_data
def load_data():
    csv_path = "gs://algo_reco/predictions/test/raw_features_dataset_with_predictions.csv"
    return pd.read_csv(csv_path)

df = load_data()

# ----------------------
# COLUMNS
# ----------------------
TARGET = "prediction_estAcceptee_bin"
PROBA = "prediction_proba_estAcceptee_bin"

MARQUE = "marqueOriginal"
TYPE_MARQUE = "typeMarqueOriginal"
PRODUIT = "libelleOriginal"
DIFF_PRIX = "DiffPrix"

# ----------------------
# TITLE
# ----------------------
st.title("📊 Dashboard Business – Acceptation des substitutions")

# ----------------------
# KPIs
# ----------------------
col1, col2, col3 = st.columns(3)

nb_transactions = len(df)
nb_produits = df[PRODUIT].nunique()
taux_acceptation = df[TARGET].mean()

col1.metric("Nombre de transactions", f"{nb_transactions}")
col2.metric("Produits originaux substitués", f"{nb_produits}")
col3.metric("Taux d’acceptation prédit", f"{taux_acceptation:.1%}")

st.divider()

# ----------------------
# ACCEPTATION PAR DIMENSIONS
# ----------------------
st.subheader("📦 Taux d’acceptation de substitution")

col1, col2, col3 = st.columns(3)

def taux_acceptation_par(col):
    return (
        df.groupby(col)[TARGET]
        .mean()
        .reset_index()
        .sort_values(TARGET, ascending=False)
    )

with col1:
    st.markdown("**Par marque originale**")
    data_marque = taux_acceptation_par(MARQUE)
    fig = px.bar(
        data_marque,
        x=TARGET,
        y=MARQUE,
        orientation="h"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Par type de marque originale**")
    data_type = taux_acceptation_par(TYPE_MARQUE)
    fig = px.bar(
        data_type,
        x=TARGET,
        y=TYPE_MARQUE,
        orientation="h"
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown("**Par type de produit**")
    data_prod = taux_acceptation_par(PRODUIT).head(15)
    fig = px.bar(
        data_prod,
        x=TARGET,
        y=PRODUIT,
        orientation="h"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------
# TOP PRODUITS ACCEPTÉS / REFUSÉS
# ----------------------
st.subheader("🏆 Produits les plus acceptés / refusés")

col1, col2 = st.columns(2)

top_acceptes = (
    df.groupby(PRODUIT)[TARGET]
    .mean()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)

top_refuses = (
    df.groupby(PRODUIT)[TARGET]
    .mean()
    .sort_values()
    .head(5)
    .reset_index()
)

with col1:
    st.markdown("**Top 5 des produits les plus acceptés**")
    fig = px.bar(
        top_acceptes,
        x=TARGET,
        y=PRODUIT,
        orientation="h"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Top 5 des produits les plus refusés**")
    fig = px.bar(
        top_refuses,
        x=TARGET,
        y=PRODUIT,
        orientation="h"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------
# ACCEPTATION VS DIFF PRIX
# ----------------------
st.subheader("💰 Taux d’acceptation en fonction de la différence de prix")

df_prix = (
    df
    .dropna(subset=[DIFF_PRIX])
    .assign(
        diff_prix_bucket=pd.cut(
            df[DIFF_PRIX],
            bins=10
        )
    )
    .groupby("diff_prix_bucket")[TARGET]
    .mean()
    .reset_index()
)

fig = px.line(
    df_prix,
    x="diff_prix_bucket",
    y=TARGET,
    markers=True
)

fig.update_layout(
    xaxis_title="Différence de prix",
    yaxis_title="Taux d’acceptation"
)

st.plotly_chart(fig, use_container_width=True)
