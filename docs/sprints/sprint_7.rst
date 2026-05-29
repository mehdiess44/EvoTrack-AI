Sprint 7 — Évaluation Fédérée Multicentrique & Optimisation Optuna
==================================================================

Le Sprint 7 correspond au notebook expérimental
``notebooks/benchmark_federated.ipynb``. Ce notebook implémente un protocole
de benchmark multicentrique pour comparer trois scénarios :

* apprentissage centralisé ;
* apprentissage fédéré standard avec FedAvg ;
* apprentissage fédéré avec FedBN, où les couches ``BatchNormalization`` restent
  locales.

.. warning::

   Ce sprint reste expérimental et académique. Il ne constitue pas une
   validation clinique réelle et ne permet pas de conclure à une performance
   diagnostique.

Objectif global du sprint
-------------------------

L'objectif réel du notebook est de valider un pipeline expérimental complet :
chargement des paires longitudinales, simulation de trois centres virtuels,
construction de datasets TensorFlow, optimisation Optuna, entraînement
centralisé, entraînement FedAvg, entraînement FedBN et visualisation finale des
courbes d'AUC.

Le notebook n'affirme pas que FedBN est définitivement supérieur. Il fournit un
cadre reproductible pour étudier si la conservation locale des statistiques
``BatchNormalization`` peut améliorer la robustesse face à un ``domain shift``
simulé.

Environnement expérimental
--------------------------

Le notebook utilise uniquement la stack suivante :

* Python ;
* TensorFlow/Keras ;
* NumPy ;
* Matplotlib ;
* Optuna ;
* ``pathlib`` et ``os`` pour les chemins et utilitaires standards.

La configuration principale est définie dans le notebook :

.. code-block:: python

   IMAGE_SIZE = (224, 224)
   BATCH_SIZE = 4
   COMMUNICATION_ROUNDS = 20
   TEST_SIZE = 0.20
   RANDOM_SEED = 42
   FAST_DEV_RUN = False
   MODEL_PATH = PROJECT_ROOT / "models" / "evotrack_siamese_best.keras"

Lorsque ``FAST_DEV_RUN`` vaut ``True``, le notebook réduit le nombre d'essais
Optuna, les rounds d'optimisation et les rounds de benchmark pour permettre un
test rapide. Dans la version actuelle, ``FAST_DEV_RUN`` est défini à ``False``
par défaut.

Données et fallback synthétique
-------------------------------

Le notebook tente d'abord de charger des paires ``T0/T1`` depuis :

.. code-block:: text

   outputs/tcia_preprocessed/images_T0
   outputs/tcia_preprocessed/images_T1

Les formats acceptés incluent ``png``, ``jpg``, ``jpeg``, ``bmp``, ``tif``,
``tiff`` et ``npy``. Les images sont chargées en niveaux de gris, redimensionnées
à ``224 x 224`` et converties en tableaux NumPy.

Si les dossiers ou les images ne sont pas disponibles, le notebook utilise
``generate_synthetic_pairs`` pour créer un petit dataset synthétique. Ce fallback
sert uniquement à vérifier que le pipeline s'exécute de bout en bout. Il ne
représente pas des données cliniques.

Labels temporaires
------------------

Lorsque les données réelles sont chargées depuis ``outputs/tcia_preprocessed``,
le notebook crée des labels temporaires équilibrés :

.. code-block:: python

   labels = np.zeros((n,), dtype=np.float32)
   labels[1::2] = 1.0

Ces labels sont explicitement provisoires. Ils permettent de calculer une AUC et
de tester le banc d'essai, mais ils ne remplacent pas des annotations cliniques
validées. Les résultats dépendent donc fortement du petit jeu de test et de ces
labels temporaires.

EPIC 1 — Partitionnement Non-IID des données cliniques
------------------------------------------------------

Le notebook applique d'abord un partitionnement train/test stratifié via
``stratified_train_test_split``. Le test global représente ``TEST_SIZE = 0.20``,
soit 20 % des données.

Les indices d'entraînement sont ensuite répartis entre trois hôpitaux virtuels
avec ``split_three_hospitals_stratified``. Cette fonction conserve autant que
possible une distribution équilibrée des classes dans ``A``, ``B`` et ``C``.

Les trois centres virtuels sont simulés ainsi :

.. list-table::
   :header-rows: 1

   * - Hôpital
     - Rôle expérimental
     - Transformation appliquée
   * - A
     - Référence
     - Aucune transformation avant prétraitement.
   * - B
     - Biais gain/offset
     - ``I' = clip(1.2 * I + 15, 0, 255)``.
   * - C
     - Bruit capteur
     - Bruit gaussien de moyenne ``0`` et ``sigma = 5``.

Prétraitement et datasets TensorFlow
------------------------------------

La fonction ``make_dataset`` applique le ``domain shift`` correspondant à
l'hôpital, convertit les images grayscale 1 canal en images 3 canaux avec
``ensure_three_channels``, puis construit un ``tf.data.Dataset``.

La fonction ``preprocess_pair`` effectue :

* resize vers ``IMAGE_SIZE = (224, 224)`` ;
* normalisation dans ``[-1, 1]`` ;
* format de sortie ``((image_T0, image_T1), label)`` pour un modèle siamois à
  deux entrées.

Les objets créés sont :

* ``ds_A`` ;
* ``ds_B`` ;
* ``ds_C`` ;
* ``test_ds``.

EPIC 2 — Modèle siamois et agrégation Keras
-------------------------------------------

Le notebook tente de charger le modèle pré-entraîné :

.. code-block:: text

   models/evotrack_siamese_best.keras

Le chargement utilise ``safe_mode=False`` et fournit la fonction custom
``absolute_difference`` via ``custom_objects``. Cette fonction est enregistrée
avec ``@tf.keras.utils.register_keras_serializable`` et calcule la différence
absolue entre les représentations ``T0`` et ``T1``.

Si le modèle réel est absent ou ne peut pas être chargé, le notebook crée un
petit modèle siamois fallback avec deux couches convolutionnelles, deux couches
``BatchNormalization``, un pooling global, une couche ``Lambda`` de différence
absolue et une sortie sigmoïde.

Les fonctions d'agrégation réellement implémentées sont :

* ``iter_leaf_layers(model)`` : parcourt récursivement les couches portant des
  poids, y compris lorsqu'elles sont imbriquées dans un sous-modèle Keras ;
* ``get_trainable_weights_dict(model)`` : retourne les poids entraînables par
  chemin de couche ;
* ``aggregate_fedavg(global_model, list_local_models)`` : moyenne les poids des
  modèles locaux et synchronise le modèle global et les modèles locaux ;
* ``aggregate_fedbn(global_model, list_local_models)`` : applique la moyenne
  aux couches non-BatchNorm et préserve les couches ``BatchNormalization``
  locales.

La contrainte FedBN est centrale : les couches détectées comme
``BatchNormalization`` ou dont le nom contient ``batch_normalization`` ne sont
pas moyennées et ne sont pas écrasées sur les modèles locaux.

EPIC 3 — Recherche d'hyperparamètres avec Optuna
------------------------------------------------

La fonction ``objective(trial)`` optimise deux hyperparamètres :

.. code-block:: python

   lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
   local_epochs = trial.suggest_categorical("local_epochs", [1, 2, 3])

Pour chaque essai, le notebook crée un modèle global et trois modèles locaux,
entraîne les modèles locaux sur ``ds_A``, ``ds_B`` et ``ds_C``, puis agrège avec
``aggregate_fedavg``. La métrique retournée à Optuna est l'AUC évaluée sur
``test_ds``.

Le nombre d'essais est contrôlé par :

.. code-block:: python

   OPTUNA_TRIALS = 2 if FAST_DEV_RUN else 15
   OPTUNA_ROUNDS = 1 if FAST_DEV_RUN else 5

L'étude est lancée avec :

.. code-block:: python

   study = optuna.create_study(direction="maximize",
                               study_name="evotrack_sprint7_fedavg")
   study.optimize(objective, n_trials=OPTUNA_TRIALS)
   best_params = study.best_params

EPIC 4 — Benchmarking des trois scénarios
-----------------------------------------

Le benchmark final utilise les meilleurs paramètres Optuna :

.. code-block:: python

   selected_lr = float(best_params.get("lr", 1e-4))
   selected_local_epochs = int(best_params.get("local_epochs", 1))

Trois scénarios sont ensuite évalués :

* ``benchmark_centralized`` concatène ``ds_A``, ``ds_B`` et ``ds_C`` dans un
  dataset global, entraîne un modèle unique et évalue l'AUC sur ``test_ds`` à
  chaque round.
* ``benchmark_federated(..., aggregate_fedavg)`` entraîne les trois modèles
  locaux, applique FedAvg et évalue le modèle global.
* ``benchmark_federated(..., aggregate_fedbn)`` utilise la même boucle, mais
  applique FedBN.

Pour chaque scénario, le notebook stocke :

* ``loss`` ;
* ``auc``.

Le nombre de pas temporels est contrôlé par :

.. code-block:: python

   BENCHMARK_ROUNDS = 2 if FAST_DEV_RUN else COMMUNICATION_ROUNDS

EPIC 5 — Visualisation et preuve expérimentale
----------------------------------------------

La visualisation finale trace les courbes d'AUC pour :

* ``Centralise`` ;
* ``FedAvg`` ;
* ``FedBN``.

L'axe X représente les rounds de communication ou les époques équivalentes.
L'axe Y représente l'AUC de validation sur ``test_ds``. La figure inclut un
titre, une grille, une légende et des couleurs distinctes.

Cette figure valide surtout la chaîne expérimentale : données, simulation
multicentrique, optimisation, entraînement et visualisation. Elle ne prouve pas
une performance clinique.

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

Si ce fichier est absent, si ``FAST_DEV_RUN`` vaut ``True`` ou si le chargement
du modèle échoue, le notebook utilise le modèle fallback minimal. Pour utiliser
les paires TCIA prétraitées, les dossiers suivants doivent être présents :

.. code-block:: text

   outputs/tcia_preprocessed/images_T0
   outputs/tcia_preprocessed/images_T1

Sinon, le fallback synthétique est utilisé.

Livrables attendus
------------------

Les livrables du notebook sont :

* les datasets ``ds_A``, ``ds_B``, ``ds_C`` et ``test_ds`` ;
* une étude Optuna et un dictionnaire ``best_params`` ;
* trois historiques ``centralized_history``, ``fedavg_history`` et
  ``fedbn_history`` ;
* les métriques ``loss`` et ``auc`` pour chaque scénario ;
* un graphique final comparant ``Centralise / FedAvg / FedBN``.

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

Note méthodologique
-------------------

Le Sprint 7 doit être lu comme un benchmark d'ingénierie et de méthode. Les
résultats dépendent du nombre de paires disponibles, du petit jeu de test, des
labels temporaires, des transformations simulées, du modèle chargé et des
hyperparamètres sélectionnés par Optuna.

Aucun résultat numérique ne doit être documenté sans exécution réelle du
notebook. Aucune conclusion clinique ne doit être tirée sans annotations
validées, protocole médical dédié, validation externe et analyse statistique
appropriée.
