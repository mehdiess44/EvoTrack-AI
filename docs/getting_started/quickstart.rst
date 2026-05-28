Démarrage rapide
================

Interface Streamlit
-------------------

L'application principale est exposée par ``app.py`` :

.. code-block:: powershell

   streamlit run app.py

Le tableau de bord permet de charger deux images longitudinales ``T0`` et
``T1``, d'exécuter le modèle siamois, d'afficher une carte de différence et de
présenter des cas similaires issus de l'index vectoriel FAISS.

Compilation locale de la documentation
--------------------------------------

.. code-block:: powershell

   sphinx-build -b html docs docs/_build/html

La page d'accueil générée se trouve ensuite dans ``docs/_build/html/index.html``.

Note d'interprétation
---------------------

Les scores, cartes de chaleur, métriques et résumés textuels sont descriptifs.
Ils ne remplacent pas une lecture médicale experte et ne doivent pas être
utilisés comme diagnostic autonome.
