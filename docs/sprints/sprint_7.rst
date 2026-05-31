Benchmark fédéré multicentrique : Centralisé vs FedAvg vs FedBN
===============================================================

Introduction
------------

Cette page documente le notebook ``notebooks/benchmark_federated.ipynb``.
Il met en place un benchmark expérimental comparant trois stratégies
d'apprentissage pour EvoTrack-AI :

* apprentissage centralisé ;
* apprentissage fédéré avec FedAvg ;
* apprentissage fédéré avec FedBN.

Le protocole simule un contexte multicentrique à partir de paires
longitudinales ``T0/T1``. Les résultats produits servent à comparer des
trajectoires expérimentales de loss et d'AUC. Ils ne constituent pas une
validation clinique et ne doivent pas être interprétés comme des performances
diagnostiques.

Objectif du benchmark
---------------------

L'objectif est de vérifier que le pipeline permet de comparer plusieurs
stratégies d'apprentissage dans un cadre multicentrique simulé. Le notebook
couvre le chargement des données, la simulation de trois hôpitaux virtuels, le
prétraitement, la construction d'un modèle siamois, l'optimisation Optuna,
l'entraînement centralisé, FedAvg, FedBN et la visualisation comparative.

Le benchmark sert donc principalement à valider la faisabilité technique du
protocole de comparaison. Il ne vise pas à démontrer une supériorité définitive
de FedBN ni à établir une conclusion clinique.

Données utilisées
-----------------

Le notebook tente de charger des paires longitudinales depuis les dossiers :

.. code-block:: text

   outputs/tcia_preprocessed/images_T0
   outputs/tcia_preprocessed/images_T1

Les formats pris en charge incluent ``png``, ``jpg``, ``jpeg``, ``bmp``,
``tif``, ``tiff`` et ``npy``. Les images sont chargées en niveaux de gris,
redimensionnées et converties en tableaux NumPy.

Si les données prétraitées ne sont pas disponibles, un fallback synthétique via
``generate_synthetic_pairs`` permet d'exécuter le notebook de bout en bout. Ce
fallback n'est pas représentatif de données cliniques.

Les labels utilisés dans ce benchmark sont temporaires et équilibrés afin de
permettre le calcul d'une AUC :

.. code-block:: python

   labels = np.zeros((n,), dtype=np.float32)
   labels[1::2] = 1.0

Ces labels ne sont pas des annotations médicales validées.

Simulation multicentrique
-------------------------

Le notebook effectue un split stratifié train/test avec ``TEST_SIZE = 0.20``.
Les données d'entraînement sont ensuite réparties entre trois hôpitaux
virtuels, également de manière stratifiée.

.. list-table::
   :header-rows: 1

   * - Hôpital
     - Rôle dans la simulation
     - Transformation
   * - A
     - Domaine de référence
     - Données proches du domaine source.
   * - B
     - Domaine avec changement d'intensité
     - Modification contrôlée du contraste ou du niveau d'intensité.
   * - C
     - Domaine bruité
     - Ajout de bruit gaussien.

Ces transformations introduisent un ``domain shift`` simulé entre centres. Ce
cadre est utile pour tester le protocole fédéré, mais il ne remplace pas une
cohorte multicentrique réelle.

Prétraitement
-------------

La fonction ``make_dataset`` construit les datasets TensorFlow associés aux
trois hôpitaux et au test global. Le prétraitement comprend :

* redimensionnement en ``224 x 224`` ;
* conversion des images grayscale vers 3 canaux si nécessaire ;
* normalisation des intensités ;
* format de sortie ``((image_T0, image_T1), label)`` pour un modèle siamois.

Les datasets principaux sont ``ds_A``, ``ds_B``, ``ds_C`` et ``test_ds``.

Modèle siamois
--------------

Le notebook tente de charger le modèle :

.. code-block:: text

   models/evotrack_siamese_best.keras

La fonction personnalisée ``absolute_difference`` est fournie au chargement du
modèle. Elle calcule la différence absolue entre les représentations des images
``T0`` et ``T1``.

Si le modèle sauvegardé ne peut pas être chargé, un modèle siamois fallback
minimal est utilisé pour garder le notebook exécutable. Ce fallback sert à
tester le protocole, pas à remplacer une expérimentation complète avec le
modèle EvoTrack-AI.

Stratégies comparées
--------------------

Centralisé
~~~~~~~~~~

Les données des hôpitaux ``A``, ``B`` et ``C`` sont concaténées dans un dataset
global. Un modèle unique est entraîné puis évalué sur ``test_ds``. Cette
stratégie sert de baseline forte, car elle a accès à toutes les données
d'entraînement.

FedAvg
~~~~~~

Chaque hôpital entraîne un modèle local sur son propre dataset. Les poids des
modèles locaux sont ensuite moyennés par ``aggregate_fedavg`` et synchronisés
avec le modèle global. FedAvg représente la baseline fédérée standard.

FedBN
~~~~~

FedBN suit le principe de FedAvg pour les couches non-BatchNorm, mais conserve
localement les couches ``BatchNormalization``. Les statistiques de
normalisation ne sont donc pas moyennées entre hôpitaux, ce qui peut être utile
lorsque les distributions locales diffèrent.

Optimisation Optuna
-------------------

Optuna sélectionne deux hyperparamètres du régime fédéré :

.. code-block:: python

   lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
   local_epochs = trial.suggest_categorical("local_epochs", [1, 2, 3])

La fonction ``objective(trial)`` exécute une mini-boucle FedAvg et retourne
l'AUC évaluée sur ``test_ds``. Le budget d'essais dépend du mode
``FAST_DEV_RUN`` :

.. code-block:: python

   OPTUNA_TRIALS = 2 if FAST_DEV_RUN else 15
   OPTUNA_ROUNDS = 1 if FAST_DEV_RUN else 5

Résultats observés
------------------

Le notebook produit trois historiques comparables :

* ``centralized_history`` ;
* ``fedavg_history`` ;
* ``fedbn_history``.

Chaque historique contient la ``loss`` et l'``auc``. Les courbes finales
montrent une comparaison expérimentale entre Centralisé, FedAvg et FedBN au fil
des rounds de communication ou des époques équivalentes.

Les performances observées restent proches. FedBN montre une stabilité
intéressante dans ce protocole, souvent autour d'un niveau d'AUC élevé, mais
FedAvg reste compétitif et le centralisé demeure une baseline forte. Les écarts
visibles sur les courbes doivent être interprétés avec prudence, notamment à
cause du faible nombre de paires et de la taille réduite du jeu de test.

Interprétation
--------------

Le centralisé sert de baseline forte, car il agrège directement toutes les
données d'entraînement dans un seul dataset. FedAvg constitue une stratégie
fédérée compétitive : la moyenne des poids reste pertinente dans cette
simulation multicentrique.

FedBN est méthodologiquement intéressant parce qu'il évite de moyenner les
statistiques Batch Normalization locales. Cette propriété peut aider face au
``domain shift`` lorsque chaque hôpital virtuel présente des distributions
d'intensité différentes. Cependant, dans cette exécution, FedBN ne démontre pas
une supériorité définitive sur FedAvg ou sur le centralisé.

Les résultats doivent donc être lus comme une validation expérimentale du
pipeline de comparaison, pas comme une preuve clinique ni comme une conclusion
généralisable sur toutes les configurations multicentriques.

Limites méthodologiques
-----------------------

Plusieurs limites encadrent l'interprétation du benchmark :

* dataset réduit ;
* jeu de test réduit ;
* labels temporaires ;
* ``domain shift`` simulé plutôt que mesuré sur plusieurs centres réels ;
* absence de validation clinique ;
* absence d'annotations médicales validées ;
* dépendance au modèle sauvegardé ou au modèle fallback ;
* budget Optuna limité.

Aucune conclusion clinique ne peut être tirée de ce notebook.

Instructions d'exécution
------------------------

Depuis la racine du projet :

.. code-block:: powershell

   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m pip install optuna jupyter
   jupyter notebook notebooks\benchmark_federated.ipynb

Exécuter ensuite le notebook cellule par cellule.

Pour utiliser le modèle réel, le fichier suivant doit être présent :

.. code-block:: text

   models/evotrack_siamese_best.keras

Pour utiliser les paires prétraitées, les dossiers suivants doivent être
présents :

.. code-block:: text

   outputs/tcia_preprocessed/images_T0
   outputs/tcia_preprocessed/images_T1

Si les données ou le modèle ne sont pas disponibles, les fallbacks du notebook
permettent de vérifier le fonctionnement général du protocole.
