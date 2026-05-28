Contrats d'interface
====================

Cette page décrit les structures de données échangées entre les principaux
modules. Les exemples sont indicatifs et issus de l'audit architectural.

Payload NLP
-----------

.. code-block:: json

   {
     "status": "stabilité | évolution",
     "surface": {
       "pixels": 1500,
       "category": "absente | microscopique | modérée | massive"
     },
     "signal": {
       "mean_intensity": 180.5,
       "category": "nulle | subtile | franche | haute intensité"
     },
     "location": {
       "description": "quadrant supérieur droit",
       "centroid": {"x": 170, "y": 40}
     },
     "safety_note": "Résumé descriptif uniquement, sans diagnostic autonome."
   }

Métriques de heatmap
--------------------

.. code-block:: json

   {
     "status": "stabilité | évolution",
     "surface_pixels": 1500,
     "surface_category": "massive",
     "intensity_mean": 180.5,
     "intensity_category": "haute intensité",
     "centroid": [170, 40],
     "location": "quadrant supérieur droit"
   }

Résultat FAISS
--------------

.. code-block:: json

   {
     "patient_id": "PT-005",
     "clinical_description": "...",
     "treatment_applied": "...",
     "outcome": "...",
     "rank": 1,
     "distance_l2": 0.4321,
     "similarity_score": 0.7839
   }

Résultat d'auto-labellisation
-----------------------------

.. code-block:: json

   {
     "patient_id": "UPENN-GBM-00001",
     "ssim_score": 0.7234,
     "label": 1,
     "t0_path": "images_T0/...",
     "t1_path": "images_T1/..."
   }

Ces contrats sont documentaires. Ils doivent être stabilisés par des tests
avant d'être considérés comme une API publique.
