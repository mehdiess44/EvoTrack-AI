Benchmark fédéré multicentrique : Centralisé vs FedAvg vs FedBN
===============================================================

Introduction
------------

Cette page documente le notebook expérimental
``notebooks/benchmark_federated.ipynb``. Il met en place un benchmark
comparatif entre trois stratégies d'entraînement pour EvoTrack-AI :

* apprentissage centralisé ;
* apprentissage fédéré avec FedAvg ;
* apprentissage fédéré avec FedBN.

Le protocole est académique et expérimental. Les métriques produites par le
notebook ne constituent pas une validation clinique et ne doivent pas être
interprétées comme des performances diagnostiques réelles.

Objectif du benchmark
---------------------

L'objectif est d'étudier l'impact d'un ``domain shift`` multicentrique simulé
sur l'entraînement d'un modèle siamois longitudinal. Trois hôpitaux virtuels
sont générés à partir des mêmes paires ``T0/T1``, avec des transformations
différentes selon le centre.

Le benchmark valide surtout le pipeline expérimental : chargement des données,
simulation multicentrique, construction des datasets TensorFlow, optimisation
Optuna, entraînement centralisé, entraînement FedAvg, entraînement FedBN et
visualisation comparative des courbes d'AUC. Il ne constitue pas une validation
clinique.

Données utilisées
-----------------

Le notebook tente d'abord de charger des paires longitudinales ``T0/T1`` depuis
les dossiers suivants :

.. code-block:: text

   outputs/tcia_preprocessed/images_T0
   outputs/tcia_preprocessed/images_T1

Les formats pris en charge incluent ``png``, ``jpg``, ``jpeg``, ``bmp``,
``tif``, ``tiff`` et ``npy``. Les images sont chargées en niveaux de gris,
redimensionnées et converties en tableaux NumPy.

Si ces données ne sont pas disponibles localement, le notebook utilise un
fallback synthétique minimal via ``generate_synthetic_pairs``. Ce fallback sert
à rendre le notebook exécutable de bout en bout, mais il ne représente pas des
données cliniques.

Lorsque les données réelles sont chargées depuis ``outputs/tcia_preprocessed``,
les labels actuels sont temporaires et équilibrés :

.. code-block:: python

   labels = np.zeros((n,), dtype=np.float32)
   labels[1::2] = 1.0

Ces labels permettent de calculer une AUC pour le benchmark, mais ils ne doivent
pas être présentés comme des annotations cliniques réelles.

Simulation multicentrique
-------------------------

Le notebook applique un partitionnement stratifié train/test, avec
``TEST_SIZE = 0.20``. Les données d'entraînement sont ensuite réparties de
manière stratifiée entre trois hôpitaux virtuels.

.. list-table::
   :header-rows: 1

   * - Hôpital
     - Description
     - Transformation
   * - A
     - Données standards
     - Aucune transformation spécifique.
   * - B
     - Biais de gain et d'offset
     - ``I' = clip(1.2 * I + 15, 0, 255)``.
   * - C
     - Données bruitées
     - Bruit gaussien avec ``sigma = 5``.

Ces transformations simulent un ``domain shift`` entre centres, par exemple des
différences de calibration ou de bruit d'acquisition.

Prétraitement
-------------

La fonction ``make_dataset`` construit les datasets TensorFlow associés aux
trois hôpitaux et au test global. Le prétraitement appliqué comprend :

* redimensionnement en ``224 x 224`` ;
* conversion des images grayscale vers 3 canaux si nécessaire ;
* normalisation dans l'intervalle ``[-1, 1]`` ;
* format de sortie ``((image_T0, image_T1), label)`` pour un modèle siamois.

Les datasets produits sont :

* ``ds_A`` ;
* ``ds_B`` ;
* ``ds_C`` ;
* ``test_ds``.

Modèle utilisé
--------------

Le notebook tente de charger le modèle pré-entraîné :

.. code-block:: text

   models/evotrack_siamese_best.keras

Le chargement fournit la fonction personnalisée ``absolute_difference`` dans
``custom_objects``. Cette fonction calcule la différence absolue entre les
représentations des deux entrées ``T0`` et ``T1``.

Si le modèle réel ne peut pas être chargé, un modèle siamois fallback minimal
est utilisé. Ce fallback sert uniquement à conserver un notebook exécutable ; il
ne remplace pas le modèle EvoTrack-AI pour une expérimentation complète.

Stratégies comparées
--------------------

Centralisé
~~~~~~~~~~

Les données des hôpitaux ``A``, ``B`` et ``C`` sont concaténées dans un seul
dataset global. Un modèle unique est entraîné puis évalué sur ``test_ds``.

FedAvg
~~~~~~

Chaque hôpital entraîne un modèle local sur son propre dataset. Les poids des
modèles locaux sont ensuite moyennés par ``aggregate_fedavg`` et synchronisés
avec le modèle global.

FedBN
~~~~~

FedBN suit le même principe que FedAvg, mais les couches
``BatchNormalization`` restent locales. Elles ne sont pas moyennées et ne sont
pas écrasées sur les modèles locaux. Cette stratégie vise à étudier l'effet de
statistiques de normalisation propres à chaque centre.

Optimisation avec Optuna
------------------------

Optuna est utilisé pour sélectionner deux hyperparamètres du régime fédéré :

.. code-block:: python

   lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
   local_epochs = trial.suggest_categorical("local_epochs", [1, 2, 3])

La fonction ``objective(trial)`` exécute une mini-boucle FedAvg et retourne
l'AUC évaluée sur ``test_ds``. Le nombre d'essais et de rounds dépend de
``FAST_DEV_RUN`` :

.. code-block:: python

   OPTUNA_TRIALS = 2 if FAST_DEV_RUN else 15
   OPTUNA_ROUNDS = 1 if FAST_DEV_RUN else 5

Résultats produits par le notebook
----------------------------------

Le notebook produit trois historiques comparables :

* ``centralized_history`` ;
* ``fedavg_history`` ;
* ``fedbn_history``.

Chaque historique contient :

* ``loss`` ;
* ``auc``.

La visualisation finale trace les courbes d'AUC de validation pour les trois
stratégies. L'axe X représente les rounds de communication ou les époques
équivalentes, et l'axe Y représente l'AUC sur ``test_ds``.

Limites méthodologiques
-----------------------

Les résultats du benchmark dépendent fortement :

* du nombre de paires disponibles ;
* de la taille du jeu de test ;
* des labels temporaires utilisés pour permettre le calcul de l'AUC ;
* du modèle chargé depuis ``models/evotrack_siamese_best.keras`` ou du fallback ;
* des transformations simulées pour les hôpitaux virtuels ;
* des hyperparamètres sélectionnés par Optuna.

Le benchmark ne prouve pas une supériorité clinique de FedBN. Il sert à valider
un protocole expérimental reproductible et à comparer le comportement relatif
de plusieurs stratégies d'entraînement dans un cadre contrôlé.

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

Si les données ne sont pas disponibles, le fallback synthétique est utilisé. Si
le modèle réel ne charge pas, le modèle fallback minimal est utilisé.

Formules utiles
---------------

Transformation de l'hôpital B :

.. math::

   I' = \operatorname{clip}(1.2I + 15,\ 0,\ 255)

Bruit gaussien de l'hôpital C :

.. math::

   I' = I + \mathcal{N}(0,\sigma^2), \quad \sigma = 5

Agrégation FedAvg :

.. math::

   \theta_{\text{global}}^{(t+1)} =
   \frac{1}{K}\sum_{k=1}^{K}\theta_k^{(t)}

Agrégation FedBN :

.. math::

   \theta_{\text{global}}^{(l,t+1)} =
   \frac{1}{K}\sum_{k=1}^{K}\theta_k^{(l,t)}
   \quad \forall l \notin \{\text{BatchNormalization}\}

Les paramètres des couches ``BatchNormalization`` restent locaux dans le
scénario FedBN.
