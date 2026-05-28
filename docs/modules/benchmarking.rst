Benchmarking
============

Les modules de benchmark mesurent des indicateurs expérimentaux. Ils ne
constituent pas une validation clinique réglementaire.

Modules concernés
-----------------

* ``clinical_benchmark.py`` : sensibilité, spécificité, PPV, NPV et intervalles
  bootstrap.
* ``system_benchmark.py`` : latence, throughput, mémoire et mesures système.

Métriques cliniques descriptives
--------------------------------

Les métriques ``Se``, ``Sp``, ``PPV`` et ``NPV`` doivent être accompagnées du
jeu de données, du protocole de sélection des cas, du seuil de décision et de
l'incertitude statistique.

Benchmark système
-----------------

Les mesures de latence et de mémoire dépendent du matériel, de l'environnement
Python, du format des images, du modèle chargé et de la disponibilité des
accélérateurs.

Prudence de communication
-------------------------

Ne pas présenter de performance clinique comme validée sans protocole
prospectif, comparaison experte, critères d'inclusion et analyse statistique
adaptée.
