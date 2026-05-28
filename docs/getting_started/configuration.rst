Configuration
=============

Configuration actuelle
----------------------

L'audit identifie plusieurs constantes encore définies dans le code :

* ``MODEL_PATH`` dans ``app.py`` pour le modèle ``.keras`` chargé par Streamlit.
* ``EVOLUTION_THRESHOLD`` dans ``app.py`` pour le seuil de décision.
* ``EMBEDDING_MODEL_NAME`` dans ``vector_search.py`` pour le modèle
  Sentence-Transformers.
* ``DEFAULT_MODEL_NAME`` dans ``clinical_summary.py`` pour le générateur de
  résumé.
* ``SSIM_THRESHOLD`` dans ``auto_curation_ssim.py`` pour la pré-annotation.
* Les seuils ECC, surface et intensité dans les modules de registration et de
  métriques.

Recommandation de configuration
-------------------------------

Pour une évolution future, ces valeurs peuvent être externalisées vers des
variables d'environnement ou un fichier ``config.yaml``. Cette documentation ne
modifie pas ce comportement afin de préserver le code métier existant.

Exemples de variables proposées :

.. code-block:: text

   EVOTRACK_MODEL_PATH=models/model.keras
   EVOTRACK_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
   EVOTRACK_NLP_MODEL=google/flan-t5-small

Ces variables sont des recommandations architecturales et non des contrats
actuellement garantis par l'application.
