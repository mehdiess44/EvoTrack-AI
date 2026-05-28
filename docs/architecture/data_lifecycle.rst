Cycle de vie des données
========================

Le flux utilisateur principal commence par deux images longitudinales et se
termine par un rendu descriptif dans Streamlit.

Étapes du pipeline
------------------

1. Chargement de ``T0`` et ``T1`` depuis l'interface Streamlit.
2. Prétraitement par redimensionnement ``224 x 224``, conversion RGB et
   normalisation dans l'intervalle ``[-1, 1]``.
3. Prédiction par le réseau siamois MobileNetV2.
4. Estimation de la direction d'évolution par différence de pixels.
5. Génération d'une carte de différence avec colormap JET.
6. Extraction de métriques descriptives : surface, intensité et centroïde.
7. Construction d'un texte clinique déterministe ou assisté par modèle NLP.
8. Recherche de cas similaires par embeddings Sentence-Transformers et FAISS.
9. Affichage du score, de l'alerte, de la carte et des cas similaires.

Représentation synthétique
--------------------------

.. code-block:: text

   Upload T0/T1
       -> preprocess_image()
       -> model.predict()
       -> compute_difference_map()
       -> extract_heatmap_metrics()
       -> build_nlp_payload()
       -> generate_clinical_summary()
       -> search_similar_cases()
       -> rendu Streamlit

Point de vigilance
------------------

La fonction ``extract_evolution_direction`` opère sur des pixels bruts sans
masquage ni normalisation robuste. L'audit documente un biais structurel vers
la classe ``Progression``. Cette limite doit être prise en compte dans toute
interprétation.
