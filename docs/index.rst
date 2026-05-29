Documentation EvoTrack-AI
=========================

EvoTrack-AI est un prototype académique d'analyse longitudinale d'images médicales.
Le projet combine un réseau siamois MobileNetV2, des cartes de différence, des
métriques descriptives, un module NLP et une recherche de cas similaires.

.. warning::

   Prototype académique : EvoTrack-AI ne constitue pas un outil de diagnostic
   médical. Les sorties doivent être interprétées comme des aides descriptives
   expérimentales, sans valeur clinique autonome.

Cette documentation est construite à partir de l'audit architectural du
2026-05-28. Elle décrit l'architecture, les contrats de données, les modules
principaux, les formules utilisées et les limites connues. Elle ne revendique
aucune performance clinique non validée.

.. toctree::
   :maxdepth: 2
   :caption: Démarrage

   getting_started/installation
   getting_started/quickstart
   getting_started/configuration

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/overview
   architecture/data_lifecycle
   architecture/interface_contracts

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/siamese_network
   modules/augmentation
   modules/inference_pipeline
   modules/nlp_module
   modules/federated_learning
   modules/benchmarking

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   sprints/sprint_7

.. toctree::
   :maxdepth: 2
   :caption: Référence

   math/formulas
   api/reference
   known_issues
