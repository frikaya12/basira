"""Point d'entrée du serveur : uvicorn basira.server:app

Variables d'environnement :
- BASIRA_DB : chemin de la base SQLite (défaut : data/basira.db)
- BASIRA_PHOTOS : dossier des photos (défaut : data/photos)
- BASIRA_MODEL : chemin d'un modèle entraîné (.pt). Sans modèle, l'IA ne donne pas d'avis.
"""

import os

from basira.api import create_app
from basira.classifier import load_classifier

app = create_app(
    db_path=os.environ.get("BASIRA_DB", "data/basira.db"),
    upload_dir=os.environ.get("BASIRA_PHOTOS", "data/photos"),
    classifier=load_classifier(os.environ.get("BASIRA_MODEL")),
)
