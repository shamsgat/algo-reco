# scripts/processing.py
import pandas as pd

def process_transactions(produits: pd.DataFrame, 
                         substitutions: pd.DataFrame, 
                         transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Merge products, substitutions, and transactions
    to create the enriched dataset with substitutions.
    """

    # Add suffix to distinguish original product columns
    produits_original = produits.add_suffix('Original')

    # Merge substitutions with original products
    substitutions_produits_original = pd.merge(
        substitutions,
        produits_original,
        on='idProduitOriginal',
        how='left'
    )

    # Add suffix for substitution products
    produits_substitution = produits.add_suffix('Substitution')

    # Merge substitutions with substitution products
    substitutions_produits_original_substitut = pd.merge(
        substitutions_produits_original,
        produits_substitution,
        on='idProduitSubstitution',
        how='left'
    )

    # Merge with transactions
    transactions_avec_substitution = pd.merge(
        transactions,
        substitutions_produits_original_substitut,
        left_on=['idProduit', 'idTransaction'],
        right_on=['idProduitOriginal', 'idTransaction'],
        how='inner'
    )

    # Encode estAcceptee as binary
    transactions_avec_substitution['estAcceptee_bin'] = (
        transactions_avec_substitution['estAcceptee'].apply(lambda x: 1 if x is False else 0)
    )

    return transactions_avec_substitution
