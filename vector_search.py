"""EvoTrack AI — Module de Case-Based Reasoning (Recherche Vectorielle).

Implémente un système de raisonnement par cas similaires pour l'aide à la
décision clinique en neuro-oncologie.  Le pipeline :

    1.  Génère une base synthétique de ~15 cas historiques de Glioblastome.
    2.  Encode les descriptions cliniques via Sentence-Transformers
        (``paraphrase-multilingual-MiniLM-L12-v2`` — 384 dims, multilingue).
    3.  Indexe les vecteurs dans un index FAISS L2 (Flat, exact search).
    4.  Expose ``search_similar_cases(query, top_k)`` pour retrouver les cas
        les plus proches d'une description générée par le système EvoTrack.

Usage autonome (validation locale) :
    python vector_search.py

Intégration :
    from vector_search import build_case_index, search_similar_cases
"""

import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================================
#  CONFIGURATION
# ============================================================================

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # Dimension de sortie du modèle MiniLM-L12


# ============================================================================
#  MODULE 1 — BASE DE DONNÉES SYNTHÉTIQUE (HISTORIQUE GBM)
# ============================================================================

SYNTHETIC_CASE_DB = [
    {
        "patient_id": "GBM-001",
        "clinical_description": "Lors de l'examen IRM de suivi précoce réalisé au temps T0+27 jours, correspondant à la phase post-chirurgicale immédiate et préalable au début de la radiothérapie, on observe un phénomène de progression précoce rapide (Rapid Early Progression). L'IRM encéphalique pondérée en T1 après injection de gadolinium révèle de multiples nouveaux nodules de rehaussement tissulaire hétérogène mesurant une taille cumulée supérieure à 15 millimètres, s'étendant le long de la marge ventriculaire droite. Le signal FLAIR indique une formation d'œdème extensif et une infiltration progressant dangereusement vers les noyaux gris centraux, confirmant une cinétique proliférative fulgurante avant même l'initiation des rayonnements ionisants.",
        "treatment_applied": "La Réunion de Concertation Pluridisciplinaire (RCP) a validé le démarrage urgent et immédiat du protocole de Stupp conventionnel, consistant en une radiothérapie externe de 60 Gy fractionnée, couplée à l'administration de Témozolomide concomitant. En raison de l'effet de masse menaçant, ce protocole a été strictement associé à une corticothérapie à haute dose par Dexaméthasone (8 mg/jour) pour gérer l'hypertension intracrânienne.",
        "outcome": "Le patient a présenté une résistance radiologique et clinique totale à la thérapie, entraînant une progression ininterrompue et un décès survenu 6,7 mois après l'intervention chirurgicale initiale. Ce cas illustre le pronostic particulièrement sombre des progressions précoces antérieures à la radio-chimiothérapie, un facteur pronostique indépendant de survie péjorative.",
    },
    {
        "patient_id": "GBM-002",
        "clinical_description": "Patiente âgée de 73 ans présentant un suivi post-opératoire. À T0+3 mois, l'IRM encéphalique multimodale démontre l'évolution d'une volumineuse masse fronto-pariétale droite caractérisée par une infiltration tissulaire traversant physiquement la brèche durale et exploitant la défectuosité osseuse issue de la craniectomie antérieure. Le signal T1 post-gadolinium confirme sans équivoque une dissémination sous-cutanée extracrânienne formant une masse de 4 centimètres dans le cuir chevelu, dotée d'un centre sévèrement nécrotique visible en hyposignal T1 et hyper-signal T2.",
        "treatment_applied": "Devant cette extension extra-neurale exceptionnelle et le statut de performance ECOG sévèrement dégradé de la patiente (indice de Karnofsky estimé à 40), l'équipe médicale a récusé toute nouvelle intervention neurochirurgicale ou radiothérapique. Des soins palliatifs exclusifs et de confort ont été instaurés après une biopsie cutanée confirmant la nature glioblastomateuse, bien que l'histologie ait démontré une adaptation phénotypique au micro-environnement sous-cutané.",
        "outcome": "La patiente a connu une progression fatale combinée au niveau local et cutané. Le décès est survenu 4,5 mois après le diagnostic de cette récidive extracrânienne atypique.",
    },
    {
        "patient_id": "GBM-003",
        "clinical_description": "Le suivi longitudinal par IRM à T0+2,5 mois révèle une croissance locale agressive au niveau du lit tumoral temporal gauche avec une prise de contraste nodulaire périphérique épaisse. Simultanément à cette évolution neurologique, une échographie abdominale, motivée par une cytolyse hépatique aiguë inexpliquée, révèle une hépatomégalie nodulaire multiple. Une biopsie hépatique échoguidée met en évidence des cellules primitives indifférenciées négatives pour les marqueurs classiques GFAP et OLIG2, mais dont l'analyse moléculaire confirme formellement qu'il s'agit d'une métastase hépatique fulgurante originaire du glioblastome primaire.",
        "treatment_applied": "Face à l'insuffisance hépatique émergente et à la maladie systémique, l'équipe d'oncologie a décidé l'interruption immédiate de l'association chimiothérapique systémique (Stupp et Bévacizumab) afin d'éviter une toxicité hépatique létale. Le patient a été orienté vers des mesures de support hépatique et métabolique strictes.",
        "outcome": "Le patient a développé une défaillance hépatique aiguë terminale, menant au décès 4,5 mois après la chirurgie encéphalique initiale. Une autopsie complète a révélé que la tumeur primaire s'était disséminée non seulement au système nerveux central, mais avait également métastasé de façon diffuse vers le système cardiovasculaire, les poumons et la moelle osseuse.",
    },
    {
        "patient_id": "GBM-004",
        "clinical_description": "Patient présentant initialement une volumineuse lésion temporo-occipitale gauche dont le diagnostic initial était ambigu, mimant un abcès parasitaire. À T0+10 jours, l'IRM d'urgence démontre un temps de doublement tumoral foudroyant, le volume tridimensionnel de la lésion passant brutalement de 30 cm³ à 60 cm³. Cette expansion s'accompagne de l'apparition de multiples altérations kystiques complexes et d'un œdème vasogénique compressif majeur entraînant un déplacement de la ligne médiane (midline shift) supérieur à 8 millimètres avec menace d'engagement temporal.",
        "treatment_applied": "Une résection chirurgicale macroscopique en urgence a été réalisée dans un but de décompression vitale. L'examen histopathologique détaillé de la pièce opératoire a définitivement confirmé le diagnostic de glioblastome de grade IV, caractérisé par une hypercellularité extrême et un index de prolifération Ki-67 dépassant les 70%, justifiant la cinétique radiologique foudroyante.",
        "outcome": "Malgré la décompression, la maladie a présenté une récidive locale extrêmement agressive au sein même de la cavité de résection sous chimiothérapie adjuvante précoce. La survie globale du patient s'est limitée à 7 mois post-diagnostic.",
    },
    {
        "patient_id": "GBM-005",
        "clinical_description": "L'IRM encéphalique de contrôle (séquences T1 avec Gadolinium et T2/FLAIR) est réalisée alors que le patient en est à son 4ème cycle de Témozolomide adjuvant. L'image vectorielle démontre un élargissement tridimensionnel de 35% de l'épaisseur de l'anneau de rehaussement temporal droit. Crucialement, le profil de la carte ADC (Apparent Diffusion Coefficient) issue de la séquence de diffusion montre une restriction très marquée de la diffusion des molécules d'eau, avec des valeurs quantitatives inférieures à 0.8 x 10^-3 mm²/s. Cette signature radiomique valide formellement une hypercellularité maligne récurrente et exclut de manière robuste l'hypothèse d'une radionécrose tissulaire.",
        "treatment_applied": "Le comité de neuro-oncologie a statué sur un diagnostic d'échec avéré du traitement de première ligne. Une thérapie de sauvetage systémique par l'anticorps monoclonal anti-VEGF Bévacizumab a été initiée à la posologie de 10 mg/kg administrée toutes les 2 semaines par voie intraveineuse.",
        "outcome": "Le patient a bénéficié d'une stabilisation clinique et radiologique transitoire d'une durée de 2 mois, suivie inéluctablement d'un échappement thérapeutique majeur se manifestant sous la forme d'une progression tumorale infiltrante et diffuse, réfractaire à toute intervention ultérieure.",
    },
    {
        "patient_id": "GBM-006",
        "clinical_description": "Patient de 60 ans atteint d'un glioblastome dont l'IRM réalisée à 8 mois post-diagnostic montre un rehaussement paramagnétique en séquence T1 irrégulier et agressif envahissant directement le splenium du corps calleux. Cette architecture en aile de papillon s'étend inexorablement au parenchyme cérébral controlatéral de l'hémisphère opposé. L'analyse volumétrique de la séquence FLAIR confirme que l'infiltrat cellulaire a triplé de volume, traduisant une migration tumorale active le long des faisceaux myélinisés profonds de la matière blanche.",
        "treatment_applied": "En l'absence stricte de toute indication chirurgicale de réduction tumorale due à l'invasion calleuse bilatérale, le patient a été inclus avec son consentement dans un essai clinique expérimental testant un protocole de rechallenge utilisant un inhibiteur de l'angiogenèse (Bévacizumab) couplé à une chimiothérapie cytotoxique de type inhibiteur de la topo-isomérase (Irinotécan).",
        "outcome": "Le patient a subi une dégradation neurologique motrice rapide caractérisée par une paraparésie évolutive et des troubles cognitifs sévères. La thérapie n'a pas permis d'infléchir la courbe de prolifération, aboutissant à une survie post-progression de seulement 3,2 mois.",
    },
    {
        "patient_id": "GBM-007",
        "clinical_description": "Ce cas documente un échec franc d'immunothérapie par inhibiteur de PD-1 (Nivolumab). L'IRM encéphalique à T0+6 semaines post-initiation de l'inhibiteur de point de contrôle montre une progression massive volumétrique des zones rehaussées (+80%) et une exacerbation spectaculaire de l'œdème sur les séquences FLAIR. Contrairement au phénomène espéré de pseudo-progression inflammatoire (évaluable par les critères iRANO), la cartographie d'hyperperfusion au PWI confirme un afflux vasculaire tumoral signant une vraie progression maligne réfractaire à la réponse immunitaire.",
        "treatment_applied": "Confrontée au risque d'engagement cérébral imminent, l'équipe médicale a procédé à l'arrêt immédiat et définitif de l'immunothérapie expérimentale. L'intervention a consisté en la réintroduction en urgence des corticostéroïdes à très forte dose (Dexaméthasone 16 mg/jour) afin de juguler la composante vasogénique de l'œdème, suivie de l'introduction d'un traitement palliatif par Bévacizumab de sauvetage.",
        "outcome": "On a constaté une absence totale de réponse physiologique au traitement de sauvetage anti-VEGF. Le décès du patient est survenu rapidement, consécutif à un engagement sous-falcoriel lié à l'effet de masse intolérable.",
    },
    {
        "patient_id": "GBM-008",
        "clinical_description": "Lésion gliale primaire de haut grade initialement traitée par l'association classique de radiothérapie et de Témozolomide. À l'échéance de 5 mois de suivi, l'IRM en coupes coronales et axiales T1 post-contraste révèle une dissémination épendymaire franche. On observe une fine prise de contraste nodulaire et linéaire qui tapisse intimement les parois des ventricules latéraux et du troisième ventricule. Aucun résidu charnu ou rehaussé massif n'est visualisé au site opératoire pariétal d'origine, mais l'analyse cytologique du liquide cérébrospinal (LCS) révèle une charge massive de cellules malignes gliales circulantes.",
        "treatment_applied": "Changement complet de paradigme thérapeutique vers une radiothérapie palliative crânio-spinale visant l'axe neural dans son intégralité, associée à une chimiothérapie de deuxième ligne par agent alkylant nitrosourée (Lomustine ou CCNU).",
        "outcome": "Le patient a développé une hydrocéphalie communicante aiguë secondaire au blocage des villosités arachnoïdiennes par les cellules tumorales, nécessitant la pose chirurgicale d'une dérivation ventriculo-péritonéale de décharge. La survie post-dissémination a été très limitée, s'établissant à 4 mois.",
    },
    {
        "patient_id": "GBM-009",
        "clinical_description": "Lors du bilan IRM à 12 mois de suivi post-opératoire, le site tumoral primaire frontal droit apparaît parfaitement stable et quiescent. Toutefois, une imagerie panmédullaire cervico-dorso-lombaire, initialement motivée par l'apparition soudaine d'une radiculopathie hyperalgique des membres inférieurs, révèle de multiples foyers de rehaussement nodulaire de la queue de cheval et de la surface du cône médullaire terminal. Cette configuration radiologique signe une dissémination leptoméningée gravissime (phénomène de drop metastasis) provenant du glioblastome sus-tentoriel primaire via la dynamique du flux du LCS.",
        "treatment_applied": "Compte tenu de la symptomatologie focale invalidante, l'équipe médicale a instauré un protocole de chimiothérapie intrathécale par ponction lombaire itérative, combiné à une radiothérapie focale de conformation sur les lésions symptomatiques volumineuses de la charnière lombo-sacrée.",
        "outcome": "Malgré la prise en charge agressive, le patient a conservé un déficit neurologique sensitivo-moteur irréversible des membres inférieurs confinant au fauteuil roulant. L'évolution systémique a mené à une survie globale de 15 mois à compter du diagnostic de la métastase.",
    },
    {
        "patient_id": "GBM-010",
        "clinical_description": "Phénotype d'échappement insidieux sous traitement chronique par Bévacizumab. L'imagerie longitudinale démontre le phénomène neuro-oncologique classique de récidive non-rehaussée typique des agents anti-angiogéniques : l'IRM T1-Gadolinium ne montre aucune nouvelle prise de contraste majeure, suggérant une barrière vasculaire faussement étanche. Cependant, l'analyse minutieuse du signal FLAIR montre une infiltration insidieuse, diffuse et bilatérale de la substance blanche, adoptant un aspect de gliomatose cérébrale infiltrante.",
        "treatment_applied": "La décision a été prise d'engager des soins de support exclusifs tout en maintenant des perfusions espacées de Bévacizumab. Ce maintien avait pour unique objectif d'éviter le phénomène rebond de perméabilité vasculaire (flare-up effect) qui complique souvent l'arrêt brutal des anti-VEGF et génère un œdème fatal en quelques jours.",
        "outcome": "La prolifération gliale infiltrante a provoqué une détérioration neurocognitive globale et progressive, menant à un état d'éveil altéré. Le décès a été prononcé 2 mois après la première documentation de cette progression radiologique FLAIR.",
    },
    {
        "patient_id": "GBM-011",
        "clinical_description": "Patient porteur d'une tumeur au profil moléculaire défavorable : IDH-sauvage et promoteur MGMT strictement non méthylé. L'IRM réalisée un mois seulement après la fin de l'irradiation fractionnée révèle un doublement volumétrique de la masse lésionnelle T1+C temporale droite. Crucialement, l'évaluation par spectroscopie par résonance magnétique (SRM) montre l'apparition d'un pic massif de lipides à 1.3 ppm, corrélé à une prolifération cellulaire anarchique dépassant les capacités d'oxygénation tissulaire, provoquant une ischémie intratumorale focale proliférative.",
        "treatment_applied": "Face à la résistance inhérente prévisible au Témozolomide (statut MGMT non méthylé) et à la progression sous traitement, la ligne chimiothérapique a été modifiée pour du Carboplatine en monothérapie intraveineuse, dans le cadre d'un protocole compassionnel.",
        "outcome": "Le protocole alternatif s'est soldé par un échec complet. La tumeur a poursuivi une évolution expansible (True Progression) entraînant une compression du tronc cérébral par engagement transtentoriel descendant fatal à 8 mois.",
    },
    {
        "patient_id": "GBM-012",
        "clinical_description": "Évolution radiologique dramatique caractérisée par l'apparition d'une lésion multifocale récidivante au temps T0+9 mois. L'IRM documente l'émergence de deux nouveaux foyers tissulaires fortement rehaussés : le premier orbito-frontal gauche mesurant 1 cm de diamètre, et le second occipital droit mesurant 1,5 cm. Ces deux lésions naissent à grande distance de la zone de radiothérapie primaire (qui avait délivré des champs de 60 Gy sur un site temporal initial), suggérant une dissémination le long des faisceaux de fibres blanches.",
        "treatment_applied": "Une approche chirurgicale individualisée a été décidée : reprise chirurgicale du foyer frontal très symptomatique (troubles du comportement et de l'inhibition), suivie d'une séance de radio-chirurgie stéréotaxique (type Gamma Knife) ciblée exclusivement sur le foyer occipital naissant.",
        "outcome": "Cette prise en charge locale agressive n'a pas pu endiguer la dissémination cellulaire. Une récidive miliaire diffuse à travers tout le parenchyme cérébral est apparue à 3 mois de la radio-chirurgie, limitant la survie globale du patient à 13,8 mois.",
    },
    {
        "patient_id": "GBM-013",
        "clinical_description": "Phénotype d'hyperprogression tumorale observé lors d'un essai de thérapie vaccinale cellulaire. À l'IRM T1+C de la 8ème semaine post-vaccination, la lésion glioblastomateuse de la fosse postérieure démontre une croissance concentrique expansive foudroyante (augmentation volumétrique calculée à 150%). Cette masse est par ailleurs compliquée par une hémorragie intratumorale aiguë spontanée, se traduisant par un hyposignal marqué sur la séquence de susceptibilité magnétique T2* et un saignement actif en phase aiguë.",
        "treatment_applied": "Face au risque imminent de compression bulbaire, une exérèse neurochirurgicale de décompression en urgence absolue a été réalisée. L'analyse histologique de la pièce opératoire a définitivement confirmé la présence d'une tumeur viable massive (90% de cellules tumorales neuro-ectodermiques actives et hyperchromatiques, pour seulement 10% de nécrose fibrinoïde), écartant de fait l'hypothèse d'une pseudo-progression d'origine inflammatoire vaccinale.",
        "outcome": "Le patient a été immédiatement retiré du protocole de l'essai clinique vaccinal. Malgré l'intervention, la dégradation de la dynamique du liquide cérébrospinal a engendré une hydrocéphalie aiguë secondaire fatale en quelques jours.",
    },
    {
        "patient_id": "GBM-014",
        "clinical_description": "Suivi clinique longitudinal d'un patient de 56 ans. À l'évaluation par IRM T1 post-Témozolomide de maintenance (après la validation de 6 cycles complets), le neuro-radiologue note une oblitération complète et symétrique des cornes frontales du système ventriculaire par un tissu bourgeonnant et intensément rehaussé par le contraste. La séquence de diffusion (DWI) révèle une restriction prononcée au sein de ce tissu épendymaire, confirmant une densité cellulaire maligne pathologique et une dissémination sous-épendymaire tapissante.",
        "treatment_applied": "L'équipe de radiothérapie a proposé une irradiation hypofractionnée (HFRT) de seconde intention, consistant en la délivrance de 35 Gy répartis en 10 fractions concentrées, dans l'objectif de contrôler le volume tumoral intraventriculaire obstructif sans dépasser la tolérance radique globale du parenchyme sain adjacent.",
        "outcome": "Bien que la toxicité radiologique aiguë ait été qualifiée de modérée et tolérable, le traitement n'a pas freiné la cinétique cellulaire. La tumeur a continué sa croissance intraventriculaire, menant à un déclin cognitif rapide et une survie totale limitée à 12 mois depuis le diagnostic initial.",
    },
    {
        "patient_id": "GBM-015",
        "clinical_description": "Cas illustrant une récidive cavitaire foudroyante. L'examen IRM de contrôle systématique à 4 mois post-chirurgie macroscopiquement complète montre une oblitération et un remplissage total de la cavité kystique de résection temporo-pariétale. L'espace mort est désormais occupé par un bourgeon tissulaire solide, lobulé, envahissant les marges, et dont la cartographie rCBV démontre une hyperperfusion sanguine chaotique traduisant un réseau vasculaire tumoral anarchique (néoangiogenèse floride).",
        "treatment_applied": "Une réintervention a été fermement refusée par le collège neurochirurgical. L'imagerie montrait un envahissement circonférentiel direct du segment M2 de l'artère cérébrale moyenne sylvienne, rendant toute exérèse associée à un risque de ramollissement ischémique hémisphérique iatrogène inacceptable. Le patient a été orienté vers un service de soins palliatifs exclusifs.",
        "outcome": "Le statut de performance a subi un déclin vertical et rapide vers un score KPS de 20 (patient grabataire, confiné au lit). Le décès est survenu à 5,5 mois après l'intervention initiale.",
    },
    {
        "patient_id": "GBM-016",
        "clinical_description": "Évaluation radiologique après 6 mois de traitement en monothérapie continue par Bévacizumab pour un glioblastome en phase de récidive confirmée (r-GBM). L'IRM cérébrale révèle une réponse complète spectaculaire selon les critères cliniques stricts RANO : effacement total de la prise de contraste nodulaire T1-Gado frontale droite, résolution complète de l'œdème vasogénique expansif en séquence FLAIR, et normalisation parfaite de l'effet de masse sur le système ventriculaire. De surcroît, le TEP-Scan aux acides aminés (FET) est strictement négatif, confirmant le silence métabolique.",
        "treatment_applied": "Forte de ces résultats, la RCP a statué sur la poursuite ininterrompue du Bévacizumab en monothérapie d'entretien toutes les deux semaines, permettant un sevrage total, progressif et réussi des corticostéroïdes (Dexaméthasone).",
        "outcome": "Le patient maintient un indice de Karnofsky à 100%, attestant de l'absence totale de symptômes neurologiques centraux ou périphériques, durant une période de 11 mois continus de suivi clinique, constituant une véritable rémission prolongée sous anti-angiogéniques.",
    },
    {
        "patient_id": "GBM-017",
        "clinical_description": "Suivi longitudinal de très long terme s'étalant sur 26 mois pour un patient atteint d'un r-GBM agressif traité par thérapie anti-VEGF exclusive. Les imageries multimodales itératives superposées (couplage IRM haute résolution et 18F-FET-PET/CT) objectivent une régression tumorale structurelle et métabolique soutenue. Les algorithmes de volumétrie calculent une diminution du volume tumoral solide de plus de 85% par rapport à l'IRM de baseline pré-Bévacizumab.",
        "treatment_applied": "Maintien ferme d'une monothérapie par Bévacizumab poursuivie avec une compliance parfaite sur 26 mois consécutifs, décision prise d'exclure toute adjonction de chimiothérapie cytotoxique classique pour préserver l'hématopoïèse du patient.",
        "outcome": "La survie sans progression (PFS) s'établit à un niveau exceptionnel dépassant largement les deux ans. Le rapport médical souligne que le patient mène une vie quotidienne indépendante et socialement active, illustrant l'existence d'une sous-population génétique extrêmement répondeuse au blocage vasculaire.",
    },
    {
        "patient_id": "GBM-018",
        "clinical_description": "Lésion primaire initialement classée IDH-sauvage avec un profil de MGMT méthylé favorable. Lors de l'apparition d'une récidive tissulaire solide en topographie périventriculaire droite, une combinaison pharmacologique de Bévacizumab et d'Irinotécan est initiée. L'IRM de contrôle à 3 mois montre une réponse partielle nette et mesurable avec une réduction du diamètre géométrique maximal de 42%. Après une période prolongée de pause thérapeutique concertée, l'IRM objective une reprise de la croissance lésionnelle.",
        "treatment_applied": "Face à cette ré-évolution, l'équipe neuro-oncologique a opté pour un rechallenge audacieux, consistant en la réintroduction stricte de la même combinaison thérapeutique (Bévacizumab + Irinotécan) qui avait préalablement fonctionné.",
        "outcome": "L'imagerie a confirmé l'obtention clinique d'une seconde réponse partielle mesurable, conférant un contrôle local itératif qui a permis de pousser la survie globale de ce patient à une durée extraordinaire de 77 mois depuis la résection neurochirurgicale initiale.",
    },
    {
        "patient_id": "GBM-019",
        "clinical_description": "Patient pédiatrique/jeune adulte souffrant d'un gliome diffus de haut grade abritant, après séquençage à haut débit (NGS), une fusion génique inattendue et rare de type NTRK3-ARHGEF7. L'IRM de baseline initiale démontrait l'existence d'une masse hémisphérique massivement envahissante et jugée chirurgicalement inopérable. L'IRM T1+C réalisée seulement 8 semaines après l'initiation d'une thérapie ciblée montre un affaissement spectaculaire du rehaussement pathologique tissulaire et une disparition de la contrainte mécanique (effet de masse) sur le tronc cérébral, constituant une réponse radiologique majeure et foudroyante.",
        "treatment_applied": "Le patient a bénéficié de l'administration orale continue de Larotrectinib, un inhibiteur hautement sélectif et spécifique du récepteur tyrosine kinase TRK. Cette approche illustre l'efficacité de la médecine de précision de type tumor agnostic, indépendante du tissu d'origine.",
        "outcome": "On note un bénéfice clinique neurologique et moteur immédiat, associé à une régression tumorale soutenue dans le temps, validant cliniquement et économiquement le criblage génomique systématique des protéines de fusion NTRK dans les tumeurs inopérables.",
    },
    {
        "patient_id": "GBM-020",
        "clinical_description": "Analyse longitudinale d'un glioblastome rare identifié porteur de la mutation activatrice BRAF V600E. L'examen IRM T1 avec contraste au troisième mois de suivi consécutif à une biopsie stéréotaxique montre une diminution volumétrique tridimensionnelle précise de 55% de la masse de la lésion thalamique droite. Cette fonte tissulaire s'accompagne d'une raréfaction drastique de la vascularisation tumorale intrinsèque, parfaitement visible sous la forme d'un effondrement du rCBV en séquence de perfusion PWI.",
        "treatment_applied": "Suite au profilage tumoral, un protocole de thérapie ciblée inhibitrice de la voie des MAP kinases a été initié, combinant quotidiennement le Dabrafenib et le Trametinib (inhibiteurs doubles BRAF/MEK).",
        "outcome": "Le patient maintient un excellent contrôle biologique de la maladie (Disease Control Rate validé comme positif). La médiane de survie sans progression (PFS) s'est étendue à 5,09 mois, un résultat inespéré pour cette topographie, le tout sans manifestation de toxicité dermatologique ou cardiaque de grade 3-4 limitant la posologie.",
    },
    {
        "patient_id": "GBM-021",
        "clinical_description": "Évaluation morphologique et fonctionnelle de la réponse immunitaire systémique suite à un protocole de vaccination par cellules dendritiques (DC). De façon paradoxale pour une thérapie réussie, l'IRM structurale à 4 mois (séquence T1 Gado) ne montre aucune différence de volume lésionnel, traduisant une stabilité morphologique stricte des marges. En revanche, l'imagerie métabolique par TEP-scan cérébral révèle une résolution complète de la captation du traceur isotopique au sein de la matrice tumorale, signant l'extinction totale et inespérée de l'activité métabolique clonale.",
        "treatment_applied": "Le schéma de traitement a consisté en des injections intradermiques mensuelles répétées de préparations vaccinales à base de cellules dendritiques allogéniques de laboratoire (souche GBM6-AD/DC) afin d'éduquer les lymphocytes T du patient.",
        "outcome": "L'analyse de survie démontre une augmentation de la survie globale de 75% par rapport aux cohortes historiques appariées. Le patient a d'ores et déjà dépassé les 28 mois de survie avec un profil immunologique périphérique riche en populations cellulaires cytotoxiques CD4+ IL17+ et CD8+.",
    },
    {
        "patient_id": "GBM-022",
        "clinical_description": "Patiente adulte rigoureusement incluse dans un essai prospectif de vaccination par cellules dendritiques autologues, préalablement maturées ex-vivo et pulsées de manière personnalisée avec des néoantigènes spécifiques de sa tumeur et des antigènes associés aux tumeurs (TAA). L'IRM encéphalique longitudinale évaluée sur une période de 12 mois objective une régression continue, progressive et très lente (phénomène de shrinkage immunologique) de la matrice tumorale solide centro-pariétale, peu à peu remplacée et circonscrite par un fin liseré cicatriciel fibro-glial protecteur.",
        "treatment_applied": "Protocole expérimental d'immunothérapie active intégrée (Vaccin DC personnalisé) sans maintien de chimiothérapie cytotoxique concomitante.",
        "outcome": "Le laboratoire d'immunologie a documenté des réponses cellulaires clonales T CD4+ et CD8+ extrêmement robustes et pérennes sur les prélèvements sanguins périphériques réguliers. La patiente reste cliniquement libre de toute progression morphologique à très long terme.",
    },
    {
        "patient_id": "GBM-023",
        "clinical_description": "Suivi d'un patient inclus dans une phase 1 d'ingénierie cellulaire pour l'injection de cellules CAR-T ciblant spécifiquement la mutation réceptrice EGFRvIII, souvent surexprimée dans les GBM. À l'évaluation du premier mois, l'IRM T1 avec injection de contraste paramagnétique indique une destruction et réduction nodulaire drastique (supérieure à 80%) de la composante tissulaire charnue rehaussée kystique située dans le lobe temporal. En parallèle, l'analyse du signal FLAIR démontre une légère augmentation transitoire de la brillance régionale, interprétée comme une inflammation neuro-immune locale bénéfique autour de la zone de lyse tumorale.",
        "treatment_applied": "Traitement reposant sur la perfusion systémique unique par voie veineuse de millions de lymphocytes T autologues génétiquement modifiés à l'aide d'un vecteur lentiviral pour exprimer le récepteur chimérique (CAR-T) EGFRvIII.",
        "outcome": "Une réponse globale objective (Objective Response Rate) a été documentée à 100% au sein de cette micro-cohorte locale. La survie sans progression est notable et soutenue, moyennant une gestion médicale stricte du risque d'œdème cérébral inflammatoire et de l'orage cytokinique modéré subséquent.",
    },
    {
        "patient_id": "GBM-024",
        "clinical_description": "Suivi évolutif d'une lésion glioblastomateuse de novo prise en charge par le dispositif crânien continu NovoTTF-100A (Tumor Treating Fields) associé de manière synergique au Bévacizumab. L'IRM à 6 mois révèle une restructuration et une consolidation physique inattendue du tissu cérébral cicatriciel, se manifestant par l'apparition localisée de micro-calcifications dystrophiques au sein des loges pariétales gauches. L'hyposignal reflétant la zone de nécrose active centrale a par ailleurs complètement disparu du parenchyme.",
        "treatment_applied": "Utilisation quotidienne continue de la monothérapie anti-VEGF associée de manière ininterrompue (plus de 18 heures par jour) à une stimulation transcrânienne par champs électriques alternatifs de fréquence intermédiaire, visant à perturber l'assemblage du fuseau mitotique tumoral.",
        "outcome": "Le dossier clinique note un maintien exceptionnel de la PFS (dépassant statistiquement la barre des 42% à 6 mois de l'essai BRAIN originel), avec une qualité de vie neuro-cognitive remarquablement préservée chez ce patient.",
    },
    {
        "patient_id": "GBM-025",
        "clinical_description": "Patient présentant une amélioration spectaculaire objectivée par la diminution drastique de l'intensité du signal de contraste d'une lésion occipitale infiltrante primitive, peu de temps après l'instauration systémique d'un inhibiteur pharmacologique de la voie MAPK (Trametinib). Cette lésion, classifiée histologiquement comme un gliome de grade IV, abritait en réalité une mutation rare sous la forme d'une fusion génique BRAF-KIAA, responsable d'une suractivation constitutive et anarchique de la prolifération cellulaire.",
        "treatment_applied": "Application stricte d'une thérapie ciblée par inhibiteur de la protéine kinase MEK1/2, prescrite en relais adjuvant après l'achèvement d'une radiothérapie focale initiale.",
        "outcome": "La réduction de la pression tumorale locale sur le cortex visuel a permis la restauration inattendue des champs visuels (l'hémianopsie latérale homonyme invalidante a été totalement résolue). L'imagerie confirme une stabilité volumétrique radiologique parfaite pendant une durée de 18 mois.",
    },
    {
        "patient_id": "GBM-026",
        "clinical_description": "Prise en charge d'un patient souffrant d'un r-GBM particulièrement volumineux et symptomatique, inclus dans un protocole de traitement expérimental associant Ipilimumab et Nivolumab. Au terme de 3 mois de double immunothérapie, l'examen IRM longitudinal décrit l'apparition d'un phénomène de cavitation tissulaire spectaculaire. Le volume tumoral solide, auparavant dense et charnu, s'est transformé en son cœur en une fine coque kystique liquidienne non rehaussante, traduction morphologique directe d'une destruction massive de la trame tumorale d'origine purement immunologique.",
        "treatment_applied": "Perfusion systémique combinée d'inhibiteurs de points de contrôle immunitaire (ICIs), ciblant simultanément les voies inhibitrices CTLA-4 et PD-1 pour lever l'anergie lymphocytaire.",
        "outcome": "Le patient a présenté une toxicité dermatologique auto-immune de grade 2, jugée parfaitement tolérable par rapport au bénéfice oncologique. La lésion crânienne est documentée comme stabilisée, kystique et cliniquement silencieuse à plus de 12 mois de recul.",
    },
    {
        "patient_id": "GBM-027",
        "clinical_description": "Évaluation d'un reliquat tumoral situé dans la région pontique chez un patient adulte, démontré stable en dimensions depuis 15 mois. Les coupes fines IRM pondérées en T1 et T2 affichent une cicatrice dont l'aspect est qualifié de fantôme, caractérisée par une rétraction du parenchyme local sans la moindre trace d'œdème vasogénique périlésionnel (le signal FLAIR est rigoureusement normalisé et identique au tissu sain).",
        "treatment_applied": "Le patient a suivi un parcours de thérapie multimodale individualisée (IMI) hautement spécialisée, incluant des séances répétées d'hyperthermie électro-modulée (mEHT) couplées à l'injection stéréotaxique de virothérapie oncolytique visant à lyser la tumeur et induire une immunité locale.",
        "outcome": "La clinique rapporte la disparition complète et définitive de l'ensemble de la symptomatologie neurologique bulbaire préexistante (dysphagie et parésie faciale). L'imagerie métabolique confirme une réponse complète.",
    },
    {
        "patient_id": "GBM-028",
        "clinical_description": "Homme adulte porteur d'un reliquat expansif tumoral logé dans le lobe temporal droit. Pris en charge sous un régime à faible dose de Bévacizumab (infusion stricte de 5 mg/kg toutes les 3 semaines), l'IRM de routine réalisée à intervalles réguliers documente un assèchement absolu de la fuite capillaire locale. Cette résolution de la rupture de la barrière hémato-encéphalique (BHE) s'opère sans aucune récidive de la formation kystique sous-jacente.",
        "treatment_applied": "Stratégie de prescription de Bévacizumab à posologie volontairement abaissée, calculée spécifiquement pour juguler l'œdème cérébral invalidant tout en limitant les risques d'hypertension artérielle sévère, de toxicité rénale tubulaire et de protéinurie souvent associés à la pleine dose.",
        "outcome": "L'analyse de cette approche montre une survie globale (OS) statistiquement prolongée chez ce patient par rapport aux historiques des dosages standards (10 mg/kg), avec un profil de tolérance d'excellente qualité préservant sa fonction rénale.",
    },
    {
        "patient_id": "GBM-029",
        "clinical_description": "Examen longitudinal d'une lésion glioblastomateuse de présentation multifocale mais porteuse d'une mutation IDH1 favorable. Après 6 mois d'un protocole chimiothérapique dense, les séquences d'imagerie T1-Gado témoignent de l'effacement de trois des quatre nodules tumoraux malins initialement détectés. Le nodule persistant résiduel s'est rétracté pour mesurer moins de 5 millimètres, n'affichant plus qu'un rehaussement punctiforme et minimaliste, inactif sur les séquences de perfusion.",
        "treatment_applied": "Administration du Témozolomide selon un schéma dit intensifié et hors standard (délivrance de 150 mg/m2 par jour pendant 5 jours consécutifs tous les cycles de 28 jours) en phase adjuvante post-chirurgicale.",
        "outcome": "Rémission partielle considérée comme très prolongée ; l'Overall Survival (OS) de ce patient a excédé le cap des 24 mois, période durant laquelle le patient a pu accomplir une réintégration socioprofessionnelle complète à temps plein.",
    },
    {
        "patient_id": "GBM-030",
        "clinical_description": "Phénomène complexe de réponse immunitaire tissulaire atypique mimant transitoirement une progression. L'IRM encéphalique post-injection d'un vaccin DC autologue révèle un élargissement temporaire du noyau central nécrotique (conséquence directe de la lyse et de la mort cellulaire immunogène - ICD) mais simultanément, un amincissement radiologique spectaculaire du rempart cellulaire actif proliférant en périphérie de la loge. La séquence de perfusion dynamique tranche formellement : le paramètre rCBV chute drastiquement de 3.0 à 1.1, prouvant le tarissement du lit vasculaire néoplasique.",
        "treatment_applied": "Thérapie par vaccin anti-cancer spécifique et actif, combinée habilement à la radio-chimiothérapie conventionnelle pour maximiser la libération d'antigènes tumoraux.",
        "outcome": "Cette évaluation radiologique complexe et subtile a finalement été validée par la RCP comme une vraie réponse thérapeutique grâce à la baisse du signal de perfusion. La Progression Free Survival (PFS) s'est étendue de manière substantielle à 16 mois sans reprise évolutive.",
    },
    {
        "patient_id": "GBM-031",
        "clinical_description": "Évaluation d'un patient porteur d'un GBM frontal dont le promoteur MGMT était classé comme fortement méthylé. L'IRM de contrôle, réalisée précisément 2 mois après la fin de la séance finale de radiothérapie fractionnée, révèle un élargissement expansif et massif (+45% de volume mesuré) du rehaussement marginal périphérique en T1, accompagné d'un œdème vasogénique floride, hypersignal en séquences T2 et FLAIR. Malgré cette image alarmante, l'examen clinique montre que le patient est paradoxalement et totalement asymptomatique, sans aucun déficit nouveau. Crucialement, l'imagerie de perfusion montre un tissu inactif avec un rCBV globalement bas (inférieur à 1.2).",
        "treatment_applied": "S'appuyant sur l'absence de symptômes et le rCBV rassurant, la RCP a pris la décision éclairée de maintenir fermement le traitement par Témozolomide adjuvant (aucune modification de la ligne thérapeutique oncologique), tout en instaurant une corticothérapie légère pour lisser l'inflammation.",
        "outcome": "Les examens IRM subséquents réalisés à 4 et 6 mois post-RT montrent un collapsus spontané, progressif et rassurant de la lésion rehaussée ainsi qu'une régression spontanée de l'œdème régional. Le patient survit avec une OS médiane mesurée de 24,5 mois, confirmant a posteriori la parfaite exactitude du diagnostic de pseudo-progression.",
    },
    {
        "patient_id": "GBM-032",
        "clinical_description": "Sur un suivi IRM effectué trois mois après une chimio-radiothérapie agressive, l'équipe radiologique note une lésion pariétale droite montrant une très forte augmentation de la prise de contraste au T1-Gado. L'aspect morphologique est irrégulier, qualifié en fromage suisse. Cependant, une imagerie fonctionnelle par TEP au traceur isotopique 11C-MET révèle un ratio d'accumulation métabolique T/N (Tumeur/Parenchyme Normal) strictement inférieur à 1,0. Cet hypométabolisme majeur de capture des acides aminés signe indéniablement la présence d'un tissu inflammatoire avasculaire en phase nécrotique, dépourvu de la machinerie protéique des cellules malignes actives.",
        "treatment_applied": "Diagnostic formel de radionécrose focale. Décision d'abstention chirurgicale complète et poursuite inébranlable du protocole thérapeutique de Stupp.",
        "outcome": "Le suivi longitudinal confirme une rétractation cicatricielle de la zone rehaussée dans l'intervalle des 6 mois suivants. Le patient poursuit son parcours de survie sans présenter la moindre détérioration neurologique centrale.",
    },
    {
        "patient_id": "GBM-033",
        "clinical_description": "Phénomène déroutant d'apparition tardive. Une évaluation de routine à 6 mois post-RT montre l'apparition d'une nouvelle zone de contraste annulaire de 18 mm dans le lit opératoire temporal, un délai qui coïncide spécifiquement avec l'activation prolongée de l'appareil par champs électriques TTFields. Malgré cette émergence, le patient demeure cliniquement asymptomatique, et la lésion ne génère aucun effet de masse critique visible sur les ventricules adjacents.",
        "treatment_applied": "Face au risque de progression vraie retardée, une biopsie chirurgicale stéréotaxique ciblée est réalisée. L'analyse certifie la nature bénigne de l'infiltrat : il s'agit d'une pure nécrose de coagulation d'origine radiologique, infiltrée par des macrophages spumeux phagocytant les débris, sans aucune cellule tumorale gliale viable détectable.",
        "outcome": "Le diagnostic de Pseudo-progression extrêmement tardive liée aux TTFields est acté. Reprise immédiate du traitement par dispositif TTFields et Témozolomide, permettant une stabilisation oncologique complète et prolongée.",
    },
    {
        "patient_id": "GBM-034",
        "clinical_description": "Suivi clinique d'une patiente de 54 ans présentant une aggravation radiologique combinée (séquences T1+C et FLAIR) 3 mois après sa chirurgie d'exérèse. L'application extrêmement stricte des critères algorithmiques RANO (qui exigent l'évaluation de la séquence T2 en l'absence rigoureuse d'augmentation de la corticothérapie) démontre que l'expansion volumétrique est très majoritairement liée à un œdème réactif du parenchyme soumis aux rayonnements ionisants, sans création d'aucune nouvelle masse nodulaire solide restreinte en diffusion.",
        "treatment_applied": "Poursuite thérapeutique standard avec pour seule modification l'adjonction d'un traitement par molécules anticonvulsivantes pour gérer et prévenir les crises comitiales focales induites par l'irritation corticale de l'œdème.",
        "outcome": "Le comité d'évaluation utilisant le standard RANO a classifié très tôt ce cas en pseudo-progression (PsP). La patiente a pu bénéficier du traitement ininterrompu et sa survie globale a atteint 19,5 mois, validant la puissance discriminative du critère.",
    },
    {
        "patient_id": "GBM-035",
        "clinical_description": "Évolution sous immunothérapie. Sous traitement par inhibiteurs de points de contrôle (ICI), la lésion glioblastomateuse temporale droite augmente brutalement et de manière inquiétante de 60% en volume tissulaire (sur séquence T1+C). Contrairement aux phénomènes inflammatoires post-radiques classiques, l'étude longitudinale de cette cohorte révèle qu'une véritable PsP sous ICI prescrits en monothérapie dans le GBM est extrêmement rare. L'application du critère iRANO permet d'offrir une période d'observation de 3 mois supplémentaires, qui a mis en exergue une prolifération tissulaire irréfutable, validant la progression.",
        "treatment_applied": "Maintien du patient sous ICI de manière temporaire durant la fenêtre d'incertitude iRANO, suivi d'un basculement d'urgence vers une chimiothérapie cytotoxique classique de sauvetage.",
        "outcome": "Le patient a subi un déclin clinique irréversible. L'évolution prouve de façon rétrospective que le rehaussement post-ICI reflétait bel et bien une maladie active foudroyante et non un afflux de lymphocytes réparateurs.",
    },
    {
        "patient_id": "GBM-036",
        "clinical_description": "Patiente se présentant aux urgences neurologiques avec des troubles cognitifs aigus et une hémi-négligence à 4 mois post-RT. L'IRM crânienne montre un œdème vasogénique massif produisant un dangereux effet de masse (avec déviation paramétrique de la ligne médiane de 4 millimètres). Un diagnostic différentiel particulièrement complexe est posé par l'équipe, car l'œdème qui accompagne typiquement la PsP est ici fortement symptomatique, violant la règle générale voulant que la PsP soit asymptomatique.",
        "treatment_applied": "L'équipe prescrit un bolus intraveineux de Dexaméthasone (12 mg/jour). L'IRM de contrôle à 48 heures révèle une disparition spectaculaire et inespérée de l'effet de masse et du volume de la loge rehaussée.",
        "outcome": "La résolution fulgurante sous simple corticoïde sans agent antitumoral confirme que la masse était un œdème inflammatoire. Ce cas valide scientifiquement que l'absence de symptômes neurologiques n'est pas un critère clinique absolu et strict pour diagnostiquer la PsP. La survie globale de la patiente s'est prolongée jusqu'à 27 mois.",
    },
    {
        "patient_id": "GBM-037",
        "clinical_description": "Étude radiologique longitudinale comparée. Tumeur glioblastomateuse de novo traitée par le protocole RT/TMZ. À 3 mois, l'IRM (séquences axiales pondérées T1 post-injection de Gadolinium) indique une augmentation suspecte, bilatérale et asymétrique des marges de la cavité, mimant une récidive bi-hémisphérique. Un contrôle strict réalisé un mois plus tard (soit à 4 mois) démontre de manière limpide que le contraste est resté parfaitement et milimétriquement stable, et que son intensité a même commencé à pâlir et à s'effacer au niveau des marges internes de la lésion.",
        "treatment_applied": "Maintien rigoureux du protocole standard de maintenance au vu de la cinétique du signal, menant à la validation rétrospective collégiale d'un événement de PsP pure.",
        "outcome": "Le patient a connu une stabilisation oncologique durable et rassurante. Ce cas souligne avec force la nécessité impérieuse de recourir à une imagerie séquentielle rapprochée pour lever l'ambiguïté spatiale des modifications tissulaires initiales.",
    },
    {
        "patient_id": "GBM-038",
        "clinical_description": "Observation d'une vaste lésion de localisation occipitale. L'imagerie post-thérapeutique détecte l'apparition d'un hypersignal bilatéral massif, expansif, mais confiné exclusivement aux séquences T2 et FLAIR, associé à une prise de contraste T1 linéaire très fine, limitée exclusivement à la lisière de la marge de résection chirurgicale. Le diagnostic pathologique de radionécrose tissulaire focale est posé grâce à l'aspect radiologique typique en feuille de fougère de l'œdème, qui s'immisce délicatement entre les sillons corticaux.",
        "treatment_applied": "Prescription audacieuse de séances d'oxygénothérapie hyperbare destinées à promouvoir la néoangiogenèse saine dans la zone irradiée, associée à un arrêt temporaire volontaire de la chimiothérapie pour évaluer librement l'évolution spontanée du site.",
        "outcome": "L'approche a engendré une réduction continue du volume FLAIR de 50% sur les 6 mois suivants, sans aucune émergence de récidive nodulaire solide (la Survie Post-Progression évaluée fut de 7.2 mois).",
    },
    {
        "patient_id": "GBM-039",
        "clinical_description": "Ce patient présente une apparente progression radiologique expansive sévère sous cycles de Témozolomide, malgré un profil moléculaire extrêmement favorable (MGMT fortement méthylé). À l'IRM de perfusion (PWI), l'algorithme d'analyse des cartes hémodynamiques montre un volume sanguin intracérébral (CBV) intratumoral littéralement effondré, confirmant que la lésion expansive correspond à une zone de nécrose radique hypoxique, causée directement par les lésions fibrinoïdes et destructrices de l'endothélium vasculaire radio-induites.",
        "treatment_applied": "Compréhension du phénomène purement radiologique vasculaire et conservation continue ininterrompue de la chimiothérapie d'entretien par Témozolomide.",
        "outcome": "L'évaluation actuarielle de survie post-progression (PPS) apporte la preuve épidémiologique d'une évolution rigoureusement similaire à celle des patients stables (catégorie nP), validant biologiquement la destruction clonogénique tumorale initiale.",
    },
    {
        "patient_id": "GBM-040",
        "clinical_description": "Jeune patiente développant au suivi une vaste lésion d'apparence extra-axiale contiguë au site d'irradiation primaire. Le signal morphologique en séquence T1 suggère fortement une progression proliférative méningée foudroyante. Toutefois, l'étude par spectroscopie de résonance magnétique démontre, sans ambiguïté aucune, une absence totale de Choline (le marqueur universel du renouvellement membranaire tumoral) couplée à un pic massif et prédominant de lipides intratumoraux. Cette signature biochimique est la traduction d'une mort cellulaire nécrotique liquéfiante post-traitement des macrophages.",
        "treatment_applied": "Une neurochirurgie de nettoyage localisé (débridement) a été exécutée afin de réduire l'effet osmotique et toxique de la nécrose liquidienne sur le parenchyme sain, sans reprise inutile d'une radiothérapie.",
        "outcome": "L'analyse histopathologique finale de la pièce excisée a confirmé avec un taux de 100% l'existence d'une radionécrose tissulaire amorphe stricte, certifiant l'éradication totale du glioblastome actif.",
    },
    {
        "patient_id": "GBM-041",
        "clinical_description": "Analyse rétrospective du dossier d'un survivant statistiquement exceptionnel (ayant franchi le cap des 10 ans post-diagnostic) souffrant d'un glioblastome primaire frontal droit de type IDH-sauvage. L'IRM longitudinale de contrôle effectuée à la huitième année de suivi montre la persistance d'une vaste cavité porencéphalique cicatricielle. Cette loge chirurgicale est tapissée d'une paroi gliale extrêmement fine, exempte de toute composante tissulaire charnue récidivante ou de foyer de néo-vascularisation hyperperfusée, malgré la présence d'une atrophie corticale régionale modérée séquellaire aux traitements.",
        "treatment_applied": "Ce patient a initialement bénéficié d'un protocole historique de résection neurochirurgicale maximale, adéquatement suivie d'une chimio-radiothérapie standard. L'anamnèse révèle des antécédents médicaux intenses de maladie de Crohn et de neuro-syphilis chronique, soulevant l'hypothèse majeure d'un profilage immunitaire inflammatoire (amorçage immun non spécifique) ayant joué un rôle antitumoral crucial.",
        "outcome": "Le score KPS du patient est remarquablement maintenu à un niveau de 90. Sa survie globale supérieure à 10 ans est un phénomène oncologique extrêmement rare, nécessitant l'intégration systémique des profils d'immunogénicité de l'hôte dans la modélisation pronostique.",
    },
    {
        "patient_id": "GBM-042",
        "clinical_description": "Évaluation neurologique d'un patient porteur d'une lésion glioblastomateuse isolée de manière tout à fait exceptionnelle au sein de la seule glande pinéale. L'imagerie par résonance magnétique répétée annuellement ne montre, de façon constante sur plus de 5 ans, aucune propagation tumorale extraglandulaire, ni de bourgeonnement, ni le moindre signe d'hydrocéphalie ventriculaire obstructive. Le radiologue souligne l'absence absolue d'infiltration microcellulaire du parenchyme thalamique directement adjacent ou de dissémination épendymaire le long de l'axe ventriculaire post-irradiation.",
        "treatment_applied": "Afin de prévenir la menace d'une hydrocéphalie, le comité a mis en œuvre une radiothérapie d'une architecture atypique, élargie de manière préventive à l'ensemble du système ventriculaire (délivrant jusqu'à 60 Gy), fermement complétée par l'agent Témozolomide. Ce traitement visait à éradiquer toute dissémination leptoméningée potentielle. Des micro-récidives ultérieures ont été stabilisées par l'anti-angiogénique Bévacizumab.",
        "outcome": "Le patient jouit d'une survie globale dépassant la barre exceptionnelle de 5 ans, à mettre en perspective avec une médiane statistique mondiale documentée de seulement 6 mois pour les tumeurs agressives de cette topographie spécifique.",
    },
    {
        "patient_id": "GBM-043",
        "clinical_description": "Chronique d'un suivi radioclinique exhaustif s'étendant sur un total de 1 235 jours. L'IRM séquentielle pondérée en T1 avec injection paramagnétique affiche, examen après examen, une parfaite et constante stabilité structurelle de la loge opératoire frontale. Une suspicion radiologique tardive et délicate entre une récidive lésionnelle nodulaire versus une modification tissulaire atrophique post-radique de novo est apparue soudainement à plus de 42 mois de suivi, géométriquement située au sein précis de l'intersection des champs de la radiothérapie antérieure croisée.",
        "treatment_applied": "Devant ce défi anatomique et balistique, et face à la complexité d'une reprise chirurgicale, la patiente a courageusement subi avec succès deux cycles espacés et complets de radiothérapie hypofractionnée (HFRT) stéréotaxique, conçus spécialement pour vaporiser ces micro-évolutions asymétriques sans brûler le cortex sain.",
        "outcome": "Maintien ferme d'une survie à long terme évaluée à plus de 42 mois. L'équipe souligne la conservation d'une bonne et autonome qualité de vie à domicile, et ce malgré un risque omniprésent et statistiquement élevé de neurotoxicité cumulative et de démence radique.",
    },
    {
        "patient_id": "GBM-044",
        "clinical_description": "Pathologie fulgurante chez une fillette de 5 ans, admise pour un Glioblastome multifocal primaire du tronc cérébral (plus précisément localisé dans le pont de Varole). L'IRM diagnostique démontre une volumineuse tumeur maligne à croissance rapide et exophytique, bourgeonnant de manière chaotique dans l'angle ponto-cérébelleux gauche, générant de graves conflits avec les nerfs crâniens. L'analyse spectroscopique (SRM) révèle un rapport délétère Cho/NAA s'élevant continuellement de mois en mois. L'architecture interne de la masse à l'imagerie est diffuse, terriblement hétérogène, et envahit le parenchyme de manière hautement asymétrique.",
        "treatment_applied": "Face à l'impossibilité technique stricte d'une résection ou même d'une biopsie (l'établissement d'un diagnostic anatomopathologique précoce et formel étant entravé par la localisation vitale interdisant tout prélèvement, sous peine de tétraplégie ou d'arrêt respiratoire immédiat), une radio-chimiothérapie palliatrice et désespérée a été initiée.",
        "outcome": "La tumeur n'a offert aucune réponse aux agents systémiques, menant à une progression neurologique létale extrêmement rapide de ce glioblastome pontin pédiatrique rare, soulignant avec amertume la limite actuelle et la complexité insurmontable du traitement chirurgical dans le territoire infratentoriel.",
    },
    {
        "patient_id": "GBM-045",
        "clinical_description": "Patient adulte se présentant aux urgences avec un tableau de déficit moteur, diagnostiqué porteur d'un glioblastome multifocal synchronisé extraordinairement vaste, touchant à la fois les compartiments anatomiques hémisphérique et infratentoriel. Les examens tomographiques (scanner) et d'imagerie par résonance magnétique avec injection intraveineuse objectivent de façon concomitante trois foyers tissulaires lointains, distincts les uns des autres. Chaque foyer se rehausse de manière hétérogène et indépendante, avec un remaniement lytique nécrotique central particulièrement étendu visible dans la lésion principale droite à cheval sur les lobes fronto-pariétaux.",
        "treatment_applied": "Une craniotomie large de sauvetage a été organisée, mais l'équipe a dû se résoudre à pratiquer l'exérèse neurochirurgicale de décompression du seul foyer principal sus-tentoriel, laissant délibérément les lésions secondaires intactes en raison de leur inaccessibilité sans séquelles gravissimes.",
        "outcome": "Le post-opératoire s'est avéré désastreux : le patient a présenté une hémorragie aiguë et massive du résidu tumoral expansif en loge opératoire, se compliquant de manière synchrone d'une embolie de l'artère pulmonaire. Il est décédé en unité de soins palliatifs terminaux seulement 11 jours post-chirurgie.",
    },
    {
        "patient_id": "GBM-046",
        "clinical_description": "Présentation déroutante d'un glioblastome de localisation exclusivement spinale primitive, centré sur la charnière cervico-dorsale C7-T1. L'IRM fine du rachis (séquences T1 et T2 à haute résolution spatiale) identifie initialement une masse solide irrégulière, de morphologie à la fois extramédullaire et intramédullaire, qui franchit allègrement le foramen osseux. L'imagerie reproduisait à la perfection la signature morphologique habituelle d'un neurofibrome tout à fait bénin, effaçant les limites propres de la moelle épinière et affleurant de manière menaçante le trajet des artères vertébrales, accompagnée d'un discret mais significatif saignement méningé sous-arachnoïdien.",
        "treatment_applied": "Face à l'aggravation clinique, les chirurgiens ont procédé à une laminectomie rachidienne décompressive par voie postérieure, assortie d'une résection partielle et prudente du tissu en cause. L'étonnement fut total lorsque le laboratoire a révélé une histologie de glioblastome primitif de grade IV confirmée, incluant une prolifération microvasculaire et une nécrose focale endothéliale caractéristique.",
        "outcome": "L'identification de ce cas de glioblastome spinal primitif rarissime, qui échappe à la littérature standard, a posé un défi diagnostique majeur et est grevé d'un pronostic neurologique évolutif globalement très péjoratif pour la fonction motrice à court terme.",
    },
    {
        "patient_id": "GBM-047",
        "clinical_description": "Bilan médullaire affichant une modification massive et longitudinale de l'intensité du signal intrinsèque de la moelle épinière, s'étendant du niveau vertébral T1 jusqu'au cône en L1. À l'exploration IRM T1 post-injection de contraste, on décrit avec précision une expansion architecturale fusiforme sévère située entre les segments T6 et T11, associée à un rehaussement médullaire pathologique franc, compact, dont les caractéristiques de signal traduisent une cellularité d'une densité tumorale extrême, enclavée dans le canal rachidien étroit.",
        "treatment_applied": "L'équipe spécialisée a déployé un protocole combiné de chimiothérapie systémique agressive et de radiothérapie focale d'urgence ciblée sur l'épicentre tumoral (T6-T11), dans la tentative désespérée de détruire l'amas cellulaire et de prévenir l'installation d'une paraplégie sensitive et motrice imminente.",
        "outcome": "Le résultat fut une détérioration fonctionnelle graduelle mais implacable. L'IRM de contrôle tardive réalisée à 17 mois montre l'établissement d'une myélomalacie (ramollissement spinal) résiduelle cicatricielle, mais surtout la progression d'une tumeur cliniquement ravageuse ayant éteint les fonctions locomotrices de la partie inférieure du corps.",
    },
    {
        "patient_id": "GBM-048",
        "clinical_description": "Observation d'une jeune patiente marocaine de 19 ans. Les investigations révèlent une vaste masse cérébelleuse primitive d'évolution sournoise. L'IRM encéphalique multiaxiale décrit une volumineuse masse occupant initialement le vermis médian et s'étendant latéralement de façon invasive à tout l'hémisphère cérébelleux droit. Le profil est hétérogène, marqué par une prise de contraste annulaire et périphérique irrégulière entourant un centre kystique liquidien et nécrotique. La localisation atypique pour cet âge a rendu le diagnostic préopératoire différentiel extrêmement délicat, incluant l'hypothèse de lésions infectieuses ou de métastases de tumeurs périphériques ignorées.",
        "treatment_applied": "L'équipe de neurochirurgie a conduit une résection chirurgicale invasive et agressive de la masse au sein de la fosse crânienne postérieure exiguë, suivie d'une consolidation par radiothérapie focale et d'un traitement systémique calqué sur le protocole adjuvant de Stupp adapté pour le compartiment sous-tentoriel.",
        "outcome": "Contrairement au sombre pronostic inhérent à cette histologie, la patiente a connu une évolution exceptionnellement et durablement favorable. Les examens cliniques soulignent l'absence totale de signe clinique ou radiologique de récidive vermienne à 12 mois de suivi neurologique rigoureux (l'étude de sa Survie Globale reste en cours avec optimisme).",
    },
    {
        "patient_id": "GBM-049",
        "clinical_description": "Identification histopathologique d'une variante tumorale rare : le glioblastome de type à petites cellules, de localisation cérébelleuse atypique. L'IRM encéphalique morphologique montre la présence d'une lésion vermienne isointense par rapport à la matière grise en séquence pondérée T1 native, qui se rehausse de manière massive, pleine et homogène après injection de Gadolinium. Le séquençage moléculaire ne révèle aucune mutation du gène H3K27 typique des gliomes de la ligne médiane habituellement observés chez l'enfant ou le jeune adulte, mais l'imagerie témoigne toutefois d'une cinétique de croissance architecturale extrêmement foudroyante.",
        "treatment_applied": "Une exérèse chirurgicale jugée macroscopiquement totale de la volumineuse lésion cérébelleuse a été réalisée sans morbidité post-opératoire majeure. Le lit tumoral a par la suite été traité par des séances de radiothérapie conformationnelle, appuyées par un traitement chimiothérapique au Témozolomide sur plusieurs cycles d'adjuvant pour prévenir la recolonisation du cervelet.",
        "outcome": "Le suivi longitudinal de ce patient dépasse dorénavant la barre des deux ans sans qu'aucun symptôme neurologique vestibulaire ou moteur, ni le moindre signe radiologique de récidive locale au niveau du tronc ou de la loge cérébelleuse ne soient détectés par les radiologues de l'équipe de recherche.",
    },
    {
        "patient_id": "GBM-050",
        "clinical_description": "Dossier médical illustrant une présentation clinique initiale hautement atypique, mimant au niveau neurologique et phénotypique une maladie dégénérative du motoneurone (Sclérose Latérale Amyotrophique), avec une faiblesse motrice invalidante, une asymétrie faciale prononcée et des troubles d'élocution sévères. Le scanner crânien réalisé en urgence, puis l'IRM multiparamétrique T1 avec gadolinium, objectivent la genèse d'une tumeur solide isolée au sein du parenchyme frontal droit. Cette lésion de taille intermédiaire est étonnamment entourée d'un œdème vasogénique extrêmement modéré et focal, suggérant un mécanisme d'interférence axonale lointaine.",
        "treatment_applied": "Le patient a rapidement été soumis à une intervention de résection chirurgicale assistée par cadre stéréotaxique, permettant l'obtention de prélèvements histologiques vitaux confirmant le diagnostic d'un Glioblastome primaire de grade IV. Un protocole thérapeutique adjuvant de radio-chimiothérapie au Témozolomide (Stupp) a immédiatement pris le relais post-opératoire.",
        "outcome": "Suite à l'ablation du foyer perturbateur, le patient a été cliniquement stabilisé et a retrouvé une fraction inespérée de ses capacités neurologiques, le maintenant en vie dans d'excellentes conditions à court et moyen terme. Ce cas souligne fermement l'importance de prescrire une IRM encéphalique sans délai face à tout tableau clinique d'allure psychiatrique, anxieuse ou neurologique atypique chez l'adulte, afin d'éliminer un diagnostic de processus expansif intracrânien asymptomatique.",
    },
]


# ============================================================================
#  MODULE 2 — EMBEDDING NLP (SENTENCE-TRANSFORMERS)
# ============================================================================

_model_cache: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Charge le modèle Sentence-Transformers (singleton, chargé une seule fois).

    Returns
    -------
    SentenceTransformer
        Modèle ``paraphrase-multilingual-MiniLM-L12-v2`` (384 dims).
    """
    global _model_cache

    if _model_cache is None:
        print(f"  ⏳ Chargement du modèle NLP : {EMBEDDING_MODEL_NAME} …")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"  ✓ Modèle NLP chargé ({EMBEDDING_DIM} dimensions)")

    return _model_cache


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode une liste de textes en vecteurs denses normalisés.

    Parameters
    ----------
    texts : list[str]
        Descriptions cliniques à encoder.

    Returns
    -------
    np.ndarray
        Matrice ``(N, 384)`` de type ``float32``, normalisée L2.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


# ============================================================================
#  MODULE 3 — INDEX FAISS (RECHERCHE VECTORIELLE)
# ============================================================================

class CaseIndex:
    """Index vectoriel FAISS pour la recherche de cas cliniques similaires.

    Encapsule :
        - L'index FAISS ``IndexFlatL2`` (recherche exacte, distance L2)
        - La référence vers la base de cas (pour restituer les métadonnées)
        - Le modèle d'embedding (chargé une seule fois)

    Parameters
    ----------
    case_db : list[dict]
        Liste de cas cliniques (chaque dict doit contenir
        ``clinical_description``).
    """

    def __init__(self, case_db: list[dict] | None = None):
        if case_db is None:
            case_db = SYNTHETIC_CASE_DB

        self.case_db = case_db
        self.index: faiss.IndexFlatL2 | None = None
        self.embeddings: np.ndarray | None = None

        self._build_index()

    def _build_index(self) -> None:
        """Encode toutes les descriptions et construit l'index FAISS."""
        descriptions = [case["clinical_description"] for case in self.case_db]

        print(f"\n  📐 Encoding de {len(descriptions)} descriptions cliniques …")
        self.embeddings = encode_texts(descriptions)

        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.index.add(self.embeddings)

        print(
            f"  ✓ Index FAISS construit — "
            f"{self.index.ntotal} vecteurs × {EMBEDDING_DIM} dims"
        )

    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        """Recherche les ``top_k`` cas les plus proches d'une requête textuelle.

        Parameters
        ----------
        query_text : str
            Description clinique générée par le système EvoTrack
            (ex : sortie de ``deterministic_summary_from_payload``).
        top_k : int
            Nombre de cas à retourner.

        Returns
        -------
        list[dict]
            Liste de dicts enrichis avec les clés supplémentaires :
            ``rank``, ``distance_l2``, ``similarity_score``.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)

        query_vec = encode_texts([query_text])
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for rank, (idx, dist) in enumerate(
            zip(indices[0], distances[0]), start=1
        ):
            if idx < 0:
                continue

            case = self.case_db[idx].copy()
            case["rank"] = rank
            case["distance_l2"] = float(dist)
            # Score de similarité cosinus approché (vecteurs normalisés L2) :
            # cos_sim ≈ 1 − d²/2
            case["similarity_score"] = float(
                max(0.0, 1.0 - dist / 2.0)
            )
            results.append(case)

        return results


# ============================================================================
#  FONCTION D'ACCÈS RAPIDE (API PUBLIQUE)
# ============================================================================

_global_index: CaseIndex | None = None


def build_case_index(case_db: list[dict] | None = None) -> CaseIndex:
    """Construit (ou retourne) l'index vectoriel global.

    Parameters
    ----------
    case_db : list[dict] or None
        Base de cas custom. Si None, utilise ``SYNTHETIC_CASE_DB``.

    Returns
    -------
    CaseIndex
        Index prêt pour la recherche.
    """
    global _global_index

    if _global_index is None or case_db is not None:
        _global_index = CaseIndex(case_db)

    return _global_index


def search_similar_cases(query_text: str, top_k: int = 3) -> list[dict]:
    """Recherche les cas cliniques les plus similaires à une requête.

    Construit l'index au premier appel (singleton), puis réutilise le cache.

    Parameters
    ----------
    query_text : str
        Description clinique à comparer (ex : sortie du système EvoTrack).
    top_k : int
        Nombre de résultats à retourner.

    Returns
    -------
    list[dict]
        Top-K cas enrichis (``rank``, ``distance_l2``, ``similarity_score``).
    """
    index = build_case_index()
    return index.search(query_text, top_k=top_k)


# ============================================================================
#  POINT D'ENTRÉE — VALIDATION LOCALE
# ============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 66)
    print("  EvoTrack AI — Module de Case-Based Reasoning (Vector Search)")
    print("  Modèle : " + EMBEDDING_MODEL_NAME)
    print("  Index  : FAISS IndexFlatL2")
    print("  Base   : " + f"{len(SYNTHETIC_CASE_DB)} cas synthétiques GBM")
    print("=" * 66)

    # ── Construction de l'index ──────────────────────────────────────────
    t0 = time.perf_counter()
    case_index = build_case_index()
    build_time = time.perf_counter() - t0
    print(f"  ⏱  Index construit en {build_time:.2f}s\n")

    # ── Tests de recherche ───────────────────────────────────────────────
    test_queries = [
        (
            "Évolution modérée localisée dans le quadrant supérieur droit, "
            "avec un signal haute intensité."
        ),
        (
            "Stabilité complète, aucune modification volumétrique significative."
        ),
        (
            "Progression rapide multifocale avec nouvelles lésions "
            "dans le corps calleux."
        ),
    ]

    for q_idx, query in enumerate(test_queries, start=1):
        print(f"{'━' * 66}")
        print(f"  🔎 Requête #{q_idx}")
        print(f"     \"{query}\"\n")

        t0 = time.perf_counter()
        results = search_similar_cases(query, top_k=3)
        search_time = time.perf_counter() - t0

        for case in results:
            sim_pct = case["similarity_score"] * 100
            print(f"  #{case['rank']}  {case['patient_id']}  "
                  f"(sim: {sim_pct:.1f}%  |  L2: {case['distance_l2']:.4f})")
            print(f"      Desc : {case['clinical_description'][:90]}…")
            print(f"      Trait: {case['treatment_applied'][:90]}…")
            print(f"      Issue: {case['outcome']}")
            print()

        print(f"  ⏱  Recherche en {search_time * 1000:.1f}ms\n")

    # ── Validation programmatique ────────────────────────────────────────
    print(f"{'━' * 66}")
    print("  ✅ Validations …")

    assert case_index.index is not None, "Index FAISS non initialisé"
    assert case_index.index.ntotal == len(SYNTHETIC_CASE_DB), \
        "Nombre de vecteurs ≠ nombre de cas"
    assert case_index.embeddings.shape == (len(SYNTHETIC_CASE_DB), EMBEDDING_DIM), \
        f"Shape embeddings incorrect : {case_index.embeddings.shape}"

    r = search_similar_cases("test", top_k=1)
    assert len(r) == 1, "Recherche top_k=1 doit retourner 1 résultat"
    assert "patient_id" in r[0], "Résultat doit contenir patient_id"
    assert "similarity_score" in r[0], "Résultat doit contenir similarity_score"
    assert 0.0 <= r[0]["similarity_score"] <= 1.0, "Score hors intervalle [0, 1]"

    print(f"  ✓ Index FAISS     : {case_index.index.ntotal} vecteurs OK")
    print(f"  ✓ Embeddings      : {case_index.embeddings.shape} OK")
    print(f"  ✓ Recherche       : top_k=1 → {r[0]['patient_id']} OK")
    print(f"  ✓ Score similarité: {r[0]['similarity_score']:.4f} ∈ [0, 1] OK")
    print(f"\n{'=' * 66}")
    print("  Tous les tests passent. Module prêt pour intégration.")
    print(f"{'=' * 66}\n")
