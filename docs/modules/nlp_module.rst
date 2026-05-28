Module NLP et recherche de cas
==============================

Le module NLP transforme les métriques en payload structuré puis en résumé
descriptif. Il fournit aussi une recherche de cas similaires.

Modules concernés
-----------------

* ``nlp_payload.py`` : normalisation des métriques et payload structuré.
* ``clinical_summary.py`` : génération de résumé en français avec fallback
  déterministe.
* ``vector_search.py`` : base synthétique de cas, embeddings et recherche
  FAISS.

Résumé clinique
---------------

Le résumé doit rester descriptif. Le champ ``safety_note`` rappelle que le
texte ne fournit pas de diagnostic autonome.

Recherche vectorielle
---------------------

``vector_search.py`` utilise le modèle
``paraphrase-multilingual-MiniLM-L12-v2`` pour produire des embeddings de
dimension 384, puis effectue une recherche L2 via FAISS. La similarité est une
approximation dérivée de la distance L2 sur vecteurs normalisés.

Limites
-------

La base de cas documentée est synthétique et codée en dur. Les cas similaires
ne doivent pas être interprétés comme des recommandations thérapeutiques
validées.
