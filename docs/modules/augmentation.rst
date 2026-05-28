Augmentation et auto-labellisation
==================================

Les modules d'augmentation servent à créer des variations d'images et à
préparer des paires pour l'entraînement expérimental.

Modules concernés
-----------------

* ``synthetic_lesions.py`` : génération de lésions ellipsoïdales synthétiques.
* ``synthetic_transforms.py`` : simulation de variations d'acquisition.
* ``auto_curation_ssim.py`` : pré-annotation par SSIM sur région d'intérêt.
* ``demo_assets.py`` : génération d'assets de démonstration.

Simulation d'acquisition
------------------------

L'audit décrit des variations photométriques combinant contraste, biais
d'intensité et bruit gaussien. Un jitter spatial indépendant peut également
être appliqué pour simuler un défaut d'alignement.

Auto-labellisation SSIM
-----------------------

``auto_curation_ssim.py`` calcule la similarité structurelle sur une bounding
box issue du masque ``T0`` dilaté. La règle documentée est :

* ``label = 0`` si ``SSIM >= 0.85`` ;
* ``label = 1`` si ``SSIM < 0.85``.

Cette règle est heuristique. Elle doit être validée sur les données ciblées
avant tout usage scientifique ou clinique.
