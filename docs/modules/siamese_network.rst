Réseau siamois
==============

Le coeur du projet est un réseau siamois basé sur MobileNetV2. Les deux
branches partagent les mêmes poids et produisent des représentations qui sont
comparées par différence absolue.

Modules concernés
-----------------

* ``src/evotrack_ai/siamese_model.py`` : construction du modèle siamois.
* ``src/evotrack_ai/siamese_generator.py`` : génération de paires d'images.
* ``src/evotrack_ai/tf_data_pipeline.py`` : pipeline TensorFlow.
* ``src/evotrack_ai/train_siamese.py`` : pré-entraînement synthétique.
* ``fine_tuning.py`` : fine-tuning défensif avec validation croisée.

Principe
--------

Le modèle calcule ``|phi(T0) - phi(T1)|`` où ``phi`` représente le backbone
MobileNetV2. La sortie est une probabilité d'évolution. Cette probabilité doit
être interprétée comme un signal expérimental, pas comme une conclusion
médicale.

Fine-tuning
-----------

Le fine-tuning décrit dans l'audit utilise une validation croisée à 5 folds,
une pénalité d'ancrage par rapport aux poids pré-entraînés et un schedule
warmup plus décroissance cosinus. Les métriques retournées par fold incluent
``accuracy``, ``f1``, ``auc`` et ``val_loss`` lorsqu'elles sont calculables.

Limites
-------

Les performances issues de l'entraînement ou du fine-tuning ne constituent pas
une validation clinique. Elles doivent être présentées avec le protocole, les
données et les intervalles d'incertitude associés.
