FROM astrocrpublic.azurecr.io/runtime:3.1-9

USER root

# Installer libgomp nécessaire pour XGBoost / LightGBM / CatBoost
RUN apt-get update && apt-get install -y libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER astro
