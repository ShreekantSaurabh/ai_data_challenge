import os

DATA_PATH = os.getenv("DATA_PATH", "customerGroups.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.pkl")
PREP_PATH = os.getenv("PREP_PATH", "models/preprocessor.pkl")
MODEL_TYPE = os.getenv("MODEL_TYPE", "ada")
GCS_URI = os.getenv("GCS_URI", None)
