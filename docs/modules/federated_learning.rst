Apprentissage fédéré
====================

Le projet contient une expérimentation d'apprentissage fédéré destinée à
explorer FedAvg, FedBN et le partitionnement de données entre clients.

Modules concernés
-----------------

* ``federated_data.py`` : partitions IID et non-IID.
* ``federated_client.py`` : entraînement local et métriques client.
* ``federated_server.py`` : agrégation globale et optimisation expérimentale.
* ``fedbn.py`` : agrégation avec exclusion des couches Batch Normalization.

FedAvg
------

FedAvg agrège les poids des clients pour mettre à jour un modèle global. Dans
l'audit, la formulation moyenne les poids des ``K`` clients.

FedBN
-----

FedBN exclut les couches Batch Normalization de l'agrégation globale afin de
conserver des statistiques locales par client.

Partitionnement non-IID
-----------------------

Le partitionnement non-IID repose sur une distribution de Dirichlet avec
``alpha = 0.5`` dans la configuration auditée. Ce choix influence fortement la
difficulté de convergence et doit être documenté dans tout rapport
expérimental.
