# EvoTrack AI

> Système hybride d'aide à la décision clinique pour le suivi longitudinal des IRM de Glioblastome. Il compare les IRM dans le temps pour détecter les évolutions tumorales via un moteur de recommandation basé sur la littérature médicale, tout en préservant la confidentialité des données.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## 📖 Documentation Officielle

L'explication complète de l'architecture, de la méthodologie scientifique et des instructions d'installation détaillées se trouve sur notre documentation en ligne.

👉 **[Consultez la documentation officielle d'EvoTrack AI](https://evotrack-ai.readthedocs.io/fr/latest/)**

## ✨ Fonctionnalités Principales

* **Vision par Ordinateur** : Suivi longitudinal et détection d'anomalies via un Réseau Siamois (MobileNetV2) avec alignement Sim2Real.
* **NLP & RAG Clinique** : Moteur de recherche vectoriel (FAISS & Sentence-Transformers) pour retrouver des cas historiques similaires.
* **Apprentissage Fédéré** : Architecture de simulation multicentrique préservant le secret médical (implémentation de FedAvg et FedBN pour contrer le Domain Shift).
* **Interface UI** : Dashboard interactif développé sous Streamlit.

## 🛠️ Stack Technique

* **Langage Principal** : Python
* **Vision & Modélisation** : TensorFlow, Keras (MobileNetV2)
* **Recherche Vectorielle & NLP** : FAISS, Sentence-Transformers
* **Interface Utilisateur** : Streamlit

## 🚀 Lancement Rapide

### Prérequis
Assurez-vous d'avoir installé les dépendances du projet :
```bash
pip install -r requirements.txt
```

### Démarrer l'Application
Lancez l'interface clinique avec Streamlit :
```bash
streamlit run app.py
```