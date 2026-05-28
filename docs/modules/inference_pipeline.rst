Pipeline d'inférence
====================

Le pipeline d'inférence assemble l'alignement, la prédiction, la carte de
différence et les métriques descriptives.

Modules concernés
-----------------

* ``registration.py`` : alignement affine ECC de ``T1`` vers ``T0``.
* ``heatmap_generator.py`` : carte de différence à partir des feature maps.
* ``metrics_extraction.py`` : surface, intensité, centroïde et localisation.
* ``longitudinal_pipeline.py`` : orchestration end-to-end.
* ``app.py`` : chemin Streamlit interactif.

Sorties principales
-------------------

Le pipeline produit un score d'évolution, une heatmap, des métriques
descriptives et un payload exploitable par le module NLP. Les cartes utilisent
un overlay JET pour visualiser les différences, avec une opacité documentée à
``alpha = 0.4``.

Interprétation
--------------

Les cartes de chaleur et les métriques ne localisent pas nécessairement une
lésion validée. Elles représentent des différences visuelles ou de features
selon les hypothèses du pipeline.

Limite critique
---------------

La direction d'évolution calculée dans ``app.py`` est connue comme biaisée vers
``Progression`` lorsqu'elle repose sur une différence brute ``T1 - T0`` non
masquée et non normalisée.
