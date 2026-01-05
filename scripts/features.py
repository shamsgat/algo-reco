# scripts/features.py
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def merge_transactions_with_substitutions( produits: pd.DataFrame, substitutions: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Merge products, substitutions, and transactions
    to create the base enriched dataset.
    """

    logger.info(
        "FEATURE ENG : Starting merge_transactions_with_substitutions | "
        "produits=%s substitutions=%s transactions=%s",
        produits.shape, substitutions.shape, transactions.shape
    )

    # Add suffix to distinguish original product columns
    produits_original = produits.add_suffix("Original")
    logger.debug("produits_original columns: %s", produits_original.columns.tolist())

    # Merge substitutions with original products
    subs_with_original = pd.merge(
        substitutions,
        produits_original,
        on="idProduitOriginal",
        how="left"
    )
    logger.info("FEATURE ENG : After merge substitutions + produits_original | shape=%s", subs_with_original.shape)

    # Add suffix for substitution products
    produits_substitution = produits.add_suffix("Substitution")
    logger.debug("produits_substitution columns: %s", produits_substitution.columns.tolist())

    # Merge substitutions with substitution products
    subs_full = pd.merge(
        subs_with_original,
        produits_substitution,
        on="idProduitSubstitution",
        how="left"
    )
    logger.info("FEATURE ENG : After merge substitution products | shape=%s", subs_full.shape)

    # Merge with transactions
    transactions_enriched = pd.merge(
        transactions,
        subs_full,
        left_on=["idProduit", "idTransaction"],
        right_on=["idProduitOriginal", "idTransaction"],
        how="inner"
    )
    logger.info("FEATURE ENG : After merge with transactions | shape=%s", transactions_enriched.shape)

    # Encode target
    transactions_enriched["estAcceptee_bin"] = (
        transactions_enriched["estAcceptee"]
        .apply(lambda x: 1 if x is False else 0)
    )

    logger.info("FEATURE ENG : Target column estAcceptee_bin created")

    return transactions_enriched


def add_similarity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add similarity and difference features between
    original product and substitution.
    """

    logger.info("FEATURE ENG : Adding similarity features | input shape=%s", df.shape)

    df = df.copy()

    df["MemeMarque"] = (
        df["marqueOriginal"] == df["marqueSubstitution"]
    ).fillna(False).astype(int)

    df["MemeNutriscore"] = (
        df["nutriscoreOriginal"] == df["nutriscoreSubstitution"]
    ).fillna(False).astype(int)
    df["MemeConditionnement"] = (
        df["conditionnementOriginal"] == df["conditionnementSubstitution"]
    ).fillna(False).astype(int)

    df["MemeTypeMarque"] = (
        df["typeMarqueOriginal"] == df["typeMarqueSubstitution"]
    ).fillna(False).astype(int)
    df["DiffPrix"] = df["prixSubstitution"] - df["prixOriginal"]

    df["MemeBio"] = (
        (df["estBioOriginal"] == True) &
        (df["estBioSubstitution"] == True)
    ).fillna(False).astype(int)

    logger.info(
        "FEATURE ENG : Similarity features added: %s",
        [
            "MemeMarque",
            "MemeNutriscore",
            "MemeConditionnement",
            "MemeTypeMarque",
            "DiffPrix",
            "MemeBio",
        ],
    )

    return df


def add_time_features(df: pd.DataFrame, date_column: str = "dateHeureTransaction") -> pd.DataFrame:
    """
    Add temporal features from transaction datetime.
    """

    logger.info("FEATURE ENG : Adding time features from column '%s'", date_column)

    df = df.copy()

    df["date"] = pd.to_datetime(df[date_column])
    df["Month"] = df["date"].dt.month
    df["Day_of_week_name"] = df["date"].dt.day_name()

    logger.info("FEATURE ENG : Time features added: Month, Day_of_week_name")

    return df


def build_features_dataset(produits: pd.DataFrame, substitutions: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline.
    """

    logger.info("FEATURE ENG : Starting full feature engineering pipeline")

    df = merge_transactions_with_substitutions(
        produits=produits,
        substitutions=substitutions,
        transactions=transactions
    )

    logger.info("FEATURE ENG : Dataset shape before feature engineering: %s", df.shape)

    df = add_similarity_features(df)
    df = add_time_features(df)

    logger.info("FEATURE ENG : Final dataset shape after feature engineering: %s", df.shape)

    return df
