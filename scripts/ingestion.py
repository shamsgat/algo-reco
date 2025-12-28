import logging
import pandas as pd
from scripts.bootstrap import init_gcp_credentials
from scripts.utils import load_data_gcs, dump_data_gcs, load_data_bq, dump_table_into_bq


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

# Initialize GCP credentials and get project ID
PROJECT_ID = init_gcp_credentials()[0]
DATASET_ID = init_gcp_credentials()[1]

# Load raw data from BigQuery to python dataframe
produits = load_data_bq(PROJECT_ID, DATASET_ID, "produits_raw")
transactions = load_data_bq(PROJECT_ID, DATASET_ID, "transactions_raw")
substitutions = load_data_bq(PROJECT_ID, DATASET_ID, "substitutions_raw")

# Data processing

produits_original = produits.add_suffix('Original')
substitutions_produits_original = pd.merge(substitutions, produits_original, \
on='idProduitOriginal' , how ='left')
produits_substitution = produits.add_suffix('Substitution')
substitutions_produits_original_substitut = pd.merge(substitutions_produits_original, produits_substitution,\
    on='idProduitSubstitution' , how ='left')

transactions_avec_substitution = pd.merge(transactions, substitutions_produits_original_substitut, \
left_on=['idProduit', 'idTransaction'], right_on=['idProduitOriginal', 'idTransaction'], how='inner')
transactions_avec_substitution['estAcceptee_bin'] = [1 if x == False else 0 for x in transactions_avec_substitution['estAcceptee']]



