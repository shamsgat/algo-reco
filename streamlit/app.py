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
    csv_path = "gs://algo_reco/predictions/test/raw_features_dataset_with_predictions_mock.csv"
    return pd.read_csv(csv_path)

df = load_data()

# ----------------------
# COLUMNS
# ----------------------
TARGET = "prediction_estAcceptee_bin"  # colonne réelle
TARGET_LABEL = "Taux d’acceptation"   # label affiché
PROBA = "prediction_proba_estAcceptee_bin"

MARQUE = "marqueOriginal"
TYPE_MARQUE = "typeMarqueOriginal"
PRODUIT = "libelleOriginal"
DIFF_PRIX = "DiffPrix"
PRODUIT_SUBST = "libelleSubstitution"

# ----------------------
# TITLE
# ----------------------
st.markdown(
    "<h1 style='text-align:center; color:#0A5F8F'>📊 Dashboard Business – Acceptation des substitutions</h1>",
    unsafe_allow_html=True
)

# ----------------------
# KPIs (cards centrées, fond léger)
# ----------------------
col1, col2, col3 = st.columns(3)

nb_transactions = len(df)
nb_produits = df[PRODUIT].nunique()
taux_acceptation = df[TARGET].mean()

def kpi(title, value, color="#0A5F8F"):
    st.markdown(
        f"""
        <div style='
            background-color: #EAF2F8;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        '>
            <div style='font-size:18px; font-weight:600; color:#0A5F8F'>{title}</div>
            <div style='font-size:32px; font-weight:700; color:{color}'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col1:
    kpi("Nombre de transactions", f"{nb_transactions}")
with col2:
    kpi("Produits originaux substitués", f"{nb_produits}")
with col3:
    kpi(TARGET_LABEL, f"{taux_acceptation:.1%}")

st.divider()

# ----------------------
# ACCEPTATION PAR DIMENSIONS
# ----------------------
st.subheader("📦 Taux d’acceptation de substitution")
col1, col2, col3 = st.columns(3)

def taux_acceptation_par(col):
    return df.groupby(col)[TARGET].mean().reset_index().sort_values(TARGET, ascending=False)

for c, label, data_col in zip([col1, col2, col3],
                              ["Marque originale", "Type de marque", "Produit"],
                              [MARQUE, TYPE_MARQUE, PRODUIT]):
    with c:
        # Titre centré
        st.markdown(f"<h4 style='text-align:center'>Par {label.lower()}</h4>", unsafe_allow_html=True)
        
        data = taux_acceptation_par(data_col)
        if data_col == PRODUIT:
            data = data.head(15)
        
        fig = px.bar(
            data,
            x=TARGET,
            y=data_col,
            orientation="h",
            labels={TARGET: TARGET_LABEL, data_col: label},
            color=TARGET,
            color_continuous_scale="Blues"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------
# TOP PRODUITS ACCEPTÉS / REFUSÉS (LISTE)
# ----------------------
st.subheader("🏆 Produits les plus acceptés / refusés")
col1, col2 = st.columns(2)

# Top produits acceptés
top_acceptes = df.groupby(PRODUIT_SUBST)[TARGET].mean().sort_values(ascending=False).head(5).reset_index()
top_acceptes = top_acceptes.rename(columns={TARGET: TARGET_LABEL, PRODUIT_SUBST: "Produit substitution"})
top_acceptes["valeur_numerique"] = top_acceptes[TARGET_LABEL]
top_acceptes[TARGET_LABEL] = top_acceptes[TARGET_LABEL].map(lambda x: f"{x:.1%}")

# Top produits refusés
top_refuses = df.groupby(PRODUIT_SUBST)[TARGET].mean().sort_values().head(5).reset_index()
top_refuses = top_refuses.rename(columns={TARGET: TARGET_LABEL, PRODUIT_SUBST: "Produit substitution"})
top_refuses["valeur_numerique"] = top_refuses[TARGET_LABEL]
top_refuses[TARGET_LABEL] = top_refuses[TARGET_LABEL].map(lambda x: f"{x:.1%}")

with col1:
    st.markdown("**Top 5 des produits les plus acceptés**")
    st.table(top_acceptes.style.background_gradient(cmap="Blues", subset=["valeur_numerique"]))
with col2:
    st.markdown("**Top 5 des produits les plus refusés**")
    st.table(top_refuses.style.background_gradient(cmap="Reds", subset=["valeur_numerique"]))

st.divider()

# ----------------------
# ACCEPTATION VS DIFF PRIX
# ----------------------
st.subheader("💰 Taux d’acceptation en fonction de la différence de prix")
df_prix = df.dropna(subset=[DIFF_PRIX]).assign(diff_prix_bucket=pd.cut(df[DIFF_PRIX], bins=10).astype(str))
df_prix = df_prix.groupby("diff_prix_bucket", as_index=False).agg(taux_acceptation=(TARGET, "mean"), volume=("diff_prix_bucket", "count"))

fig = px.bar(
    df_prix,
    x="diff_prix_bucket",
    y="taux_acceptation",
    text=df_prix["taux_acceptation"].map(lambda x: f"{x:.1%}"),
    labels={"taux_acceptation": TARGET_LABEL, "diff_prix_bucket": "Différence de prix (€)"},
    color="taux_acceptation",
    color_continuous_scale="Teal"
)
fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# HEATMAP INTERACTIVE CARRÉE
# ----------------------
st.subheader("🔥 Heatmap interactive : Marque originale vs Marque de substitution")
pivot_marque = df.pivot_table(index="marqueOriginal", columns="marqueSubstitution", values=TARGET, aggfunc="mean").fillna(0)
pivot_long = pivot_marque.reset_index().melt(id_vars="marqueOriginal", var_name="marqueSubstitution", value_name=TARGET_LABEL)

fig = px.density_heatmap(
    pivot_long,
    x="marqueSubstitution",
    y="marqueOriginal",
    z=TARGET_LABEL,
    color_continuous_scale="Blues",
    text_auto=True,
    labels={"marqueOriginal": "Marque originale", "marqueSubstitution": "Marque de substitution"},
    hover_data={TARGET_LABEL: ':.1%'}
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_layout(height=800, width=800, coloraxis_colorbar=dict(title=TARGET_LABEL))
st.plotly_chart(fig, use_container_width=True)
