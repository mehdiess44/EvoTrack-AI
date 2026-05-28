Limites connues
===============

Statut académique
-----------------

EvoTrack-AI est un prototype académique. Il ne constitue pas un outil de
diagnostic médical et ne doit pas être utilisé pour prendre une décision
clinique autonome.

Biais de direction d'évolution
------------------------------

L'audit documente un bug non résolu dans ``extract_evolution_direction``. La
fonction calcule :

.. math::

   \operatorname{diff} = I_{T_1} - I_{T_0}

puis compare la somme des différences positives à la somme absolue des
différences négatives. Ce calcul est effectué sur pixels bruts, sans masque de
lésion, sans normalisation robuste et sans correction explicite des variations
d'acquisition.

Conséquence : la fonction est structurellement biaisée vers ``Progression``.
Les résultats qui dépendent de cette direction doivent être considérés comme
fragiles.

Dette technique documentée
--------------------------

* Plusieurs fonctions critiques ont des annotations de types manquantes ou
  incomplètes.
* Certaines docstrings ne décrivent pas formellement les paramètres et retours.
* La gestion d'erreurs est limitée sur les images corrompues, les datasets
  absents et les lectures de fichiers.
* Plusieurs constantes sont codées en dur et devraient être externalisées.
* ``src/evotrack_ai/__init__.py`` ne définit pas d'exports publics.
* Aucun répertoire ``tests/`` n'a été identifié dans l'audit.

Communication des résultats
---------------------------

Les métriques expérimentales doivent être accompagnées du protocole, du jeu de
données, du seuil de décision et des limites statistiques. Aucune performance
clinique ne doit être présentée comme validée sans étude dédiée.
