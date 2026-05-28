Vue d'ensemble
==============

EvoTrack-AI est organisé en couches fonctionnelles : données, augmentation,
entraînement, apprentissage fédéré, inférence, NLP et interface.

Cartographie des couches
------------------------

.. list-table::
   :header-rows: 1

   * - Couche
     - Modules principaux
     - Responsabilité
   * - Interface
     - ``app.py``
     - Tableau de bord Streamlit et orchestration utilisateur.
   * - NLP et recherche
     - ``clinical_summary.py``, ``nlp_payload.py``, ``vector_search.py``
     - Résumés descriptifs, payloads structurés et recherche FAISS.
   * - Modèle
     - ``siamese_model.py``, ``train_siamese.py``, ``fine_tuning.py``
     - Réseau siamois MobileNetV2, pré-entraînement et fine-tuning.
   * - Inférence
     - ``longitudinal_pipeline.py``, ``heatmap_generator.py``,
       ``metrics_extraction.py``, ``registration.py``
     - Alignement, cartes de différence et métriques descriptives.
   * - Augmentation
     - ``synthetic_lesions.py``, ``synthetic_transforms.py``,
       ``auto_curation_ssim.py``
     - Données synthétiques, variations d'acquisition et pré-annotation SSIM.
   * - Fédéré
     - ``federated_client.py``, ``federated_server.py``,
       ``federated_data.py``, ``fedbn.py``
     - Expérimentation FedAvg, FedBN et partitions IID/non-IID.
   * - Benchmark
     - ``clinical_benchmark.py``, ``system_benchmark.py``
     - Métriques expérimentales et mesures système.

Flux architectural simplifié
----------------------------

.. code-block:: text

   Données -> Augmentation -> Entraînement -> Modèle .keras
      |                                      |
      v                                      v
   Prétraitement -> Registration -> Heatmap -> Métriques -> NLP
      |                                                   |
      v                                                   v
   Interface Streamlit <------------------------ Recherche FAISS

L'architecture est expérimentale. Les modules sont conçus pour explorer des
chaînes d'analyse longitudinales, non pour constituer un dispositif médical.
