Formules
========

Réseau siamois
--------------

.. math::

   P(Y=1) = \sigma\left(W_2 \cdot \operatorname{ReLU}\left(
   W_1 \cdot \operatorname{GAP}\left(|\phi(T_0) - \phi(T_1)|\right) + b_1
   \right) + b_2\right)

.. math::

   \Delta\phi = |\phi(T_0) - \phi(T_1)| \in \mathbb{R}^{7 \times 7 \times 1280}

Normalisation MobileNetV2
-------------------------

.. math::

   x_{\text{norm}} = \frac{x_{\text{uint8}}}{127.5} - 1.0 \in [-1, 1]

Fine-tuning défensif
--------------------

.. math::

   \mathcal{L}_{\text{total}} =
   \operatorname{BCE}(y, \hat{y}) +
   \lambda \sum_i \|\theta_i - \theta_i^{\text{pre}}\|^2

.. math::

   \operatorname{lr}(t) =
   \begin{cases}
   \operatorname{lr}_{\text{peak}} \times \frac{t}{t_{\text{warmup}}}
   & \text{si } t < t_{\text{warmup}} \\
   \frac{\operatorname{lr}_{\text{peak}}}{2}
   \left(1 + \cos\left(\pi \times
   \frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\right)\right)
   & \text{sinon}
   \end{cases}

SSIM
----

.. math::

   \operatorname{SSIM}(x, y) =
   \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}
   {(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

Registration ECC
----------------

.. math::

   \operatorname{ECC} =
   \frac{\|\hat{I}_0^T \cdot \hat{I}_1\|}
   {\|\hat{I}_0\| \cdot \|\hat{I}_1\|}

Heatmap
-------

.. math::

   D(x,y) = \frac{1}{C}\sum_{c=1}^{C}
   \left|\phi_c^{T_1}(x,y) - \phi_c^{T_0}(x,y)\right|

.. math::

   O = (1 - \alpha) \cdot I_{\text{base}} + \alpha \cdot H_{\text{JET}},
   \quad \alpha = 0.4

Recherche vectorielle
---------------------

.. math::

   d_{L_2}(q, v) = \|q - v\|_2^2

.. math::

   \operatorname{sim} = \max\left(0,\; 1 - \frac{d_{L_2}}{2}\right)

Apprentissage fédéré
--------------------

.. math::

   \theta_{\text{global}}^{(t+1)} =
   \frac{1}{K}\sum_{k=1}^{K}\theta_k^{(t)}

.. math::

   \theta_{\text{global}}^{(l,t+1)} =
   \frac{1}{K}\sum_{k=1}^{K}\theta_k^{(l,t)}
   \quad \forall l \notin \{\text{BN layers}\}

Métriques de benchmark
----------------------

.. math::

   \operatorname{Se} = \frac{TP}{TP + FN}
   \qquad
   \operatorname{Sp} = \frac{TN}{TN + FP}

.. math::

   \operatorname{PPV} = \frac{TP}{TP + FP}
   \qquad
   \operatorname{NPV} = \frac{TN}{TN + FN}

Direction d'évolution
---------------------

.. math::

   \operatorname{diff} = I_{T_1} - I_{T_0}

.. math::

   \operatorname{growth} =
   \sum_{p:\operatorname{diff}(p)>0}\operatorname{diff}(p),
   \quad
   \operatorname{shrink} =
   \sum_{p:\operatorname{diff}(p)<0}|\operatorname{diff}(p)|

Cette dernière règle est documentée comme problématique lorsqu'elle est
appliquée à des pixels bruts sans masquage ni normalisation.
