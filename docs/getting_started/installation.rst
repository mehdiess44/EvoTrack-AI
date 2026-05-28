Installation
============

Prérequis
---------

Le projet cible un environnement Python moderne compatible avec TensorFlow,
Streamlit et les bibliothèques de vision par ordinateur utilisées par le
pipeline.

Installation de l'application
-----------------------------

Depuis la racine du dépôt :

.. code-block:: powershell

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

La dépendance ``protobuf`` doit rester contrainte à ``>=3.20.0,<5.0.0`` pour
préserver la compatibilité Streamlit documentée dans l'audit.

Installation de la documentation
--------------------------------

La documentation Sphinx possède ses propres dépendances :

.. code-block:: powershell

   python -m pip install -r docs/requirements.txt

.. warning::

   Prototype académique : EvoTrack-AI ne constitue pas un outil de diagnostic
   médical. L'installation ne fournit aucune garantie de performance clinique.
