# ============================================================================
# SUPPRESSION DES AVERTISSEMENTS
# ============================================================================
# Certains modules peuvent générer des warnings (avertissements) lors de
# l'exécution. Cette section les désactive pour une sortie plus propre.
# ============================================================================
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# IMPORTS DES BIBLIOTHÈQUES NÉCESSAIRES
# ============================================================================
# Cette section regroupe toutes les bibliothèques utilisées dans le script.
# Chaque import est accompagné d’une courte explication de son rôle.
# ============================================================================

from openai import OpenAI
# 📦 Bibliothèque officielle d'OpenAI qui permet de communiquer avec diverses API de LLM
# Utilisée ici pour se connecter à OpenRouter, une plateforme qui agrège plusieurs modèles IA

import tiktoken
# 🔢 Outil développé par OpenAI pour compter précisément le nombre de tokens
# (un token ≈ 0.75 mots en anglais ou 0.5 mots en français)
# Utile pour estimer les coûts et la complexité de traitement des modèles

import time
# ⏱️ Module standard Python pour mesurer le temps d'exécution
# Permet de calculer la durée de réponse de chaque modèle

import json
# 🗂️ Module standard Python pour manipuler des données au format JSON
# Utilisé pour sauvegarder les résultats des comparaisons dans un fichier lisible

from datetime import datetime
# 📅 Module standard Python pour gérer les dates et heures
# Permet d’horodater les exécutions et de créer des noms de fichiers uniques

# ============================================================================
# 🧠 TRAITEMENT DU TEXTE ET ANALYSE LINGUISTIQUE
# ============================================================================
# Ces bibliothèques servent à analyser la qualité linguistique et la structure
# des réponses produites par les modèles (phrases, mots, lisibilité, etc.)
# ============================================================================

import spacy
# 🧩 Bibliothèque de NLP (Natural Language Processing) très performante
# Permet de faire de l’analyse grammaticale, de la reconnaissance d’entités,
# ou encore de découper un texte en phrases ou en mots.

import textstat
# 📊 Bibliothèque dédiée à l’évaluation de la lisibilité des textes
# Permet de calculer des scores comme le Flesch Reading Ease,
# c’est-à-dire la facilité de compréhension d’un texte

import re
# ✂️ On utilise regex pour tokenizer le texte en mots 

# ============================================================================
# ⚙️ INITIALISATION DE SPACY
# ============================================================================
# Le modèle linguistique "en_core_web_sm" doit être téléchargé au préalable avec :
# 👉  python -m spacy download en_core_web_sm
# Ce modèle contient les données nécessaires à l’analyse du texte en anglais.
# ============================================================================
import subprocess
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("🧠 Modèle SpaCy 'en_core_web_sm' non trouvé. Téléchargement en cours...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")  # rechargement après téléchargement

# Chargement du modèle linguistique anglais pour SpaCy
# (Utilisé pour traiter les réponses textuelles des modèles)

# ============================================================================
# IMPORTS POUR LE TRAITEMENT DES DONNÉES ET LE CHARGEMENT DU MODÈLE
# ============================================================================
# Cette section regroupe les bibliothèques nécessaires pour :
#   1️⃣ Charger un modèle machine learning sauvegardé (.pkl ou joblib)
#   2️⃣ Manipuler et préparer les données pour la prédiction
# ============================================================================

import pickle
# 📦 Module standard Python pour sérialiser et désérialiser des objets Python
# Permet de charger un modèle sauvegardé au format .pkl
# ⚠️ Attention aux problèmes de compatibilité entre versions Python ou librairies

import joblib
# 🗂️ Alternative plus robuste à pickle, recommandée pour les modèles scikit-learn volumineux
# Permet de charger des modèles RandomForest, SVM, pipelines, etc.

import pandas as pd
# 🐼 Bibliothèque pour manipuler facilement les tableaux de données (DataFrame)
# Permet de convertir des dictionnaires/lists en DataFrame pour l'analyse et la prédiction

import numpy as np
# 🔢 Bibliothèque pour manipuler efficacement des tableaux numériques (array)
# Utile si les données doivent être converties en format numpy avant d’être passées au modèle

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

import markdown

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ============================================================================
# CONFIGURATION DU CLIENT API
# ============================================================================

# Initialisation du client pour communiquer avec OpenRouter
# OpenRouter est un service qui permet d'accéder à plusieurs modèles de LLM
# (GPT, Claude, Gemini, Llama, etc.) via une seule interface unifiée
client = OpenAI(
    # URL de base de l'API OpenRouter (différente de l'API OpenAI classique)
    base_url="https://openrouter.ai/api/v1",
    
    # Clé API personnelle pour authentifier les requêtes
    # ⚠️ ATTENTION : Cette clé doit rester confidentielle !
    api_key=os.getenv("api_key")
)

# ============================================================================
# CATALOGUE DES MODÈLES
# ============================================================================

# Liste des modèles d'intelligence artificielle (LLM)
# disponibles pour les tests. Chaque modèle appartient à une entreprise ou un
# laboratoire différent, et possède des performances et vitesses variées.
MODELES_DISPONIBLES = {
    # 🧠 Modèles de Meta (LLaMA)
    "meta-llama/llama-3.1-8b-instruct" : 8000000000.0,
    "meta-llama/llama-3.1-70b-instruct" : 70000000000.0,

    # 🤖 Modèles d’OpenAI (ChatGPT)
    "openai/gpt-4.1-nano" : 8000000000.0,
    "openai/gpt-4-turbo" : 1800000000000.0,

    # 🌐 Modèles de Google (Gemini)
    "google/gemini-2.5-flash" : 5000000000.0,
    "google/gemini-2.0-flash-001" : 5000000000.0,

    # 🧩 Modèles d’Anthropic (Claude)
    "anthropic/claude-3-haiku" : 20000000000.0,
    "anthropic/claude-3.5-sonnet" : 200000000000.0,

    # 🛰️ Modèle de X (anciennement Twitter, par Elon Musk)
    "x-ai/grok-4-fast" : 314000000000.0,

    # 🇫🇷 Modèle de Mistral AI (open source européen)
    "mistralai/mistral-7b-instruct" : 7000000000.0,

    # 🔬 Modèle de DeepSeek (open source asiatique)
    "deepseek/deepseek-r1-0528-qwen3-8b" : 8000000000.0
}

# ============================================================================
# FONCTIONS PRINCIPALES
# ============================================================================

# Fonctions de base utilisées pour :
#   1️⃣ Interroger un modèle d’intelligence artificielle (LLM)
#   2️⃣ Compter le nombre de tokens (unités de texte) dans une réponse

def interroger_modele(nom_modele: str, prompt: str, temperature: float = 0.7) -> str:
    """
    Envoie une requête à un modèle LLM et renvoie sa réponse textuelle.

    Args:
        nom_modele (str): Nom complet du modèle à utiliser
            ➜ Exemple : "openai/gpt-4-turbo"
        prompt (str): Texte ou question à envoyer au modèle
            ➜ Exemple : "Explique ce qu’est un transformeur en 3 phrases simples."
        temperature (float): Contrôle la créativité du modèle
            ➜ 0.0 = réponses très factuelles et stables
            ➜ 1.0 = réponses plus libres et créatives

    Returns:
        str: La réponse textuelle générée par le modèle
    """
    try:
        # Envoi de la requête au modèle via le client API
        # - 'model' : identifiant du modèle à interroger
        # - 'messages' : format du dialogue (ici, un seul message utilisateur)
        # - 'temperature' : influence sur la créativité de la réponse
        reponse = client.chat.completions.create(
            model=nom_modele,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

        # Extraction du texte de la première réponse du modèle
        return reponse.choices[0].message.content

    except Exception as erreur:
        # En cas de problème (ex : modèle indisponible, erreur réseau...),
        # on renvoie un message d’erreur clair au lieu de planter le programme.
        return f"❌ Erreur : {str(erreur)}"

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Compte le nombre de tokens dans un texte donné.
    Un token est une unité de texte (mot, morceau de mot ou symbole) utilisée
    par les modèles de langage pour comprendre et traiter les phrases.

    Args:
        text (str): Le texte dont on veut connaître la taille en tokens.
        model (str): Le modèle utilisé pour déterminer l’encodage (par défaut "gpt-4").

    Returns:
        int: Le nombre total de tokens dans le texte.
    """
    # Récupère le schéma d’encodage spécifique au modèle choisi
    encoding = tiktoken.encoding_for_model(model)

    # Convertit le texte en une liste de tokens numériques
    tokens = encoding.encode(text)

    # Renvoie simplement la longueur de cette liste (le nombre de tokens)
    return len(tokens)

# ============================================================================
# EXEMPLE D'UTILISATION : COMPARER TOUS LES MODÈLES
# ============================================================================

# Cette section exécute le script complet lorsqu’il est lancé directement.
# Elle permet de comparer plusieurs modèles LLM entre eux sur une même question.
# Le script mesure :
#   ✅ Le temps de réponse
#   ✅ Le nombre de tokens générés
#   ✅ Le contenu de la réponse
# Et sauvegarde le tout dans un fichier JSON pour analyse.

# 🔹 Fonction pour calculer l'empreinte CO2 en fonction du modèle
def calcul_co2(consommation_kwh, modele_name):
    """
    Retourne l'empreinte carbone (kg CO2) estimée en fonction
    de la consommation électrique et du modèle.
    
    Args:
        consommation_kwh (float): Consommation d'énergie prédite en kWh
        modele_name (str): Nom du modèle utilisé
        
    Returns:
        float: Empreinte CO2 estimée en kg
    """
    
    co2_emission_per_kwh_us = 0.233  # kg CO2 par kWh (valeur moyenne us)
    
    modele_name_lower = modele_name.lower()

    # 🧠 Modèles Meta (LLaMA)
    if "llama-3.1-8b" in modele_name_lower:
        facteur_multiplicateur = co2_emission_per_kwh_us
        empreinte_carbone_de_l_entrainement_min = 75000
        empreinte_carbone_de_l_entrainement_max = 100000
        nombre_moyen_de_requêtes_min = 1 * 10**6
        nombre_moyen_de_requêtes_max = 10 * 10**6
        
    elif "llama-3.1-70b" in modele_name_lower:
        facteur_multiplicateur = co2_emission_per_kwh_us
        empreinte_carbone_de_l_entrainement_min = 122500
        empreinte_carbone_de_l_entrainement_max = 203000
        nombre_moyen_de_requêtes_min = 1 * 10**6
        nombre_moyen_de_requêtes_max = 10 * 10**6

    # 🤖 Modèles OpenAI (GPT)
    elif "gpt-4.1-nano" in modele_name_lower:
        facteur_multiplicateur = co2_emission_per_kwh_us
        empreinte_carbone_de_l_entrainement_min = 5000000
        empreinte_carbone_de_l_entrainement_max = 10000000
        nombre_moyen_de_requêtes_min = 50 * 10**6
        nombre_moyen_de_requêtes_max = 100 * 10**6
        
    elif "gpt-4-turbo" in modele_name_lower:
        facteur_multiplicateur = co2_emission_per_kwh_us
        empreinte_carbone_de_l_entrainement_min = 15000000
        empreinte_carbone_de_l_entrainement_max = 25000000
        nombre_moyen_de_requêtes_min = 100 * 10**6
        nombre_moyen_de_requêtes_max = 200 * 10**6

    # 🌐 Modèles Google (Gemini)
    elif "gemini-2.5-flash" in modele_name_lower:
        facteur_multiplicateur = co2_emission_per_kwh_us
        empreinte_carbone_de_l_entrainement_min = 2000000
        empreinte_carbone_de_l_entrainement_max = 4000000
        nombre_moyen_de_requêtes_min = 20 * 10**6
        nombre_moyen_de_requêtes_max = 50 * 10**6
        
    elif "gemini-2.0-flash-001" in modele_name_lower:
        facteur_multiplicateur = co2_emission_per_kwh_us
        empreinte_carbone_de_l_entrainement_min = 1800000
        empreinte_carbone_de_l_entrainement_max = 4000000
        nombre_moyen_de_requêtes_min = 20 * 10**6
        nombre_moyen_de_requêtes_max = 60 * 10**6

    # 🧩 Modèles Anthropic (Claude)
    elif "claude-3-haiku" in modele_name_lower:
        facteur_multiplicateur = co2_emission_per_kwh_us
        empreinte_carbone_de_l_entrainement_min = 200000
        empreinte_carbone_de_l_entrainement_max = 400000
        nombre_moyen_de_requêtes_min = 20 * 10**6
        nombre_moyen_de_requêtes_max = 50 * 10**6
        
    elif "claude-3.5-sonnet" in modele_name_lower:
        facteur_multiplicateur = co2_emission_per_kwh_us
        empreinte_carbone_de_l_entrainement_min = 1000000
        empreinte_carbone_de_l_entrainement_max = 2000000
        nombre_moyen_de_requêtes_min = 40 * 10**6
        nombre_moyen_de_requêtes_max = 80 * 10**6

    # 🛰️ X-AI (Grok)
    elif "grok-4-fast" in modele_name_lower:
        facteur_multiplicateur = co2_emission_per_kwh_us
        empreinte_carbone_de_l_entrainement_min = 3000000
        empreinte_carbone_de_l_entrainement_max = 6000000
        nombre_moyen_de_requêtes_min = 10 * 10**6
        nombre_moyen_de_requêtes_max = 30 * 10**6

    # 🇫🇷 Mistral AI
    elif "mistral-7b-instruct" in modele_name_lower:
        facteur_multiplicateur = co2_emission_per_kwh_us
        empreinte_carbone_de_l_entrainement_min = 350000
        empreinte_carbone_de_l_entrainement_max = 600000
        nombre_moyen_de_requêtes_min = 5 * 10**6
        nombre_moyen_de_requêtes_max = 15 * 10**6

    # 🔬 DeepSeek
    elif "deepseek-r1-0528-qwen3-8b" in modele_name_lower:
        facteur_multiplicateur = co2_emission_per_kwh_us
        empreinte_carbone_de_l_entrainement_min = 500000
        empreinte_carbone_de_l_entrainement_max = 900000
        nombre_moyen_de_requêtes_min = 5 * 10**6
        nombre_moyen_de_requêtes_max = 20 * 10**6

    # 🔹 Par défaut si le modèle n'est pas reconnu
    else:
        facteur_multiplicateur = 1.0
        empreinte_carbone_de_l_entrainement_min = 0
        empreinte_carbone_de_l_entrainement_max = 0
        nombre_moyen_de_requêtes_min = 1
        nombre_moyen_de_requêtes_max = 1

    # Empreinte carbone estimée
    return [consommation_kwh * facteur_multiplicateur, empreinte_carbone_de_l_entrainement_min / nombre_moyen_de_requêtes_max, empreinte_carbone_de_l_entrainement_max / nombre_moyen_de_requêtes_min]


def request_llm_co2_consumption(question, id_modele):

    nb_param = MODELES_DISPONIBLES[id_modele]

    # 🗂️ Dictionnaire pour stocker tous les résultats
    resultats = {
        "question": question,                            # La question commune posée
        "date_execution": datetime.now().isoformat(),    # Date et heure d’exécution au format ISO
        "modeles": []                                    # Liste qui contiendra les résultats par modèle
    }

    # 🕒 On mesure le temps de début pour calculer la durée de réponse
    debut = time.time()

    # 📤 On envoie la question au modèle et on récupère sa réponse
    reponse = interroger_modele(id_modele, question)

    # ⏱️ On calcule le temps écoulé entre l’envoi et la réception de la réponse
    duree = time.time() - debut

    # 🔢 On compte le nombre de tokens contenus dans la réponse
    nombre_tokens = count_tokens(reponse)
    
    # 🔍 Analyse du texte avec SpaCy (part-of-speech tagging, segmentation...)
    doc = nlp(question)

    # 📊 Comptage du nombre de mots par catégorie grammaticale (POS = Part of Speech)
    # Exemple : combien d’adjectifs, de verbes, de noms...
    pos_counts = doc.count_by(spacy.attrs.POS)

    # ✂️ Découpage du texte en mots avec RegEx
    def word_tokenize(text: str):
        pattern = r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?|\d+(?:[\.,]\d+)*|[^\w\s]"
        return re.findall(pattern, text)

    words = word_tokenize(question)

    # 🧾 On enregistre les résultats obtenus pour ce modèle
    resultats["modeles"].append({
        "modele_name": id_modele,                                         # Nom du modèle testé
        "reponse": reponse,                                               # Réponse textuelle générée
        "response_token_length": nombre_tokens,                           # Nombre total de tokens dans la réponse
        "response_duration": round(duree, 2),                             # Durée de génération (en secondes)
        "model_size": nb_param,                                           # Nombre de paramètres du modèle
        "adj_count": pos_counts.get(spacy.symbols.ADJ, 0),                # Nombre d’adjectifs dans la réponse
        "polysyllabcount": textstat.polysyllabcount(question),            # Nombre de mots polysyllabiques (3 syllabes ou plus)
        "long_word_count": sum(1 for word in words if len(word) > 6)      # Nombre de mots "longs" (plus de 6 lettres)
    })

    # ========================================================================
    # 💾 EXPORT DES RÉSULTATS
    # ========================================================================
    # On crée un nom de fichier unique contenant la date et l’heure d’exécution.
    # Exemple : ./output/resultats_llm_20251106_103200.json
    # ========================================================================

    output_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    nom_fichier = os.path.join(output_dir, f"resultats_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # 📦 On écrit le dictionnaire des résultats dans un fichier JSON
    # - ensure_ascii=False permet de conserver les accents
    # - indent=2 rend le fichier lisible avec une indentation propre
    with open(nom_fichier, 'w', encoding='utf-8') as fichier:
        json.dump(resultats, fichier, ensure_ascii=False, indent=2)
        
    # ============================================================================
    # 🔹 PRÉPARATION DES DONNÉES ET PREDICTION AVEC LE MODELE
    # ============================================================================

    # Cette section transforme les résultats des modèles en DataFrame,
    # charge le modèle ML sauvegardé, effectue les prédictions, et ajoute
    # les résultats prédits dans le DataFrame.

    # 1️⃣ Conversion de la liste de dictionnaires en DataFrame pandas
    # resultats["modeles"] contient les informations de chaque modèle (réponse, métriques, etc.)
    df_resultats = pd.DataFrame(resultats["modeles"])

    # 2️⃣ Chargement du modèle machine learning sauvegardé
    # Ici, joblib est utilisé pour charger un modèle RandomForest ou similaire
        
    model_path = os.path.join(BASE_DIR, "random_forest.pkl")
    modele = joblib.load(model_path)

    # 3️⃣ Préparation des données pour la prédiction
    # - On supprime les colonnes non numériques ou non pertinentes pour le modèle
    #   ici 'modele_name' et 'reponse'
    # - inplace=False crée une copie, sans modifier df_resultats original
    features = df_resultats.drop(columns=['modele_name', 'reponse'], inplace=False)

    # 4️⃣ Faire la prédiction avec le modèle sur les données préparées
    prediction = modele.predict(features)

    # 5️⃣ Ajouter les prédictions dans le DataFrame
    # - Nouvelle colonne "energy_consumption (kWh)" qui contient la valeur prédite
    df_resultats["energy_consumption (kWh)"] = prediction

    # 6️⃣ Afficher les 10 premières lignes du DataFrame final
    # - Permet de vérifier que les prédictions ont bien été ajoutées
    df_resultats.head(10)

    # 🔹 Création de la colonne CO2
    df_resultats[["co2_emission_request", "co2_emission_train_model_min", "co2_emission_train_model_max"]] = pd.DataFrame(
        df_resultats.apply(
            lambda row: calcul_co2(row["energy_consumption (kWh)"], row["modele_name"]),
            axis=1
        ).tolist(),
        index=df_resultats.index
    )

    # 🔹 Vérification
    return df_resultats


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', models=MODELES_DISPONIBLES)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json    
    question = data.get('prompt', '')
    id_modele = data.get('model', 'google/gemini-2.5-flash')
    
    results = request_llm_co2_consumption(question, id_modele)
    return jsonify({
        "response": markdown.markdown(results.loc[0, 'reponse']),
        "model": results.loc[0, 'modele_name'],
        "co2_emission_request": float(results.loc[0, 'co2_emission_request']),
        "co2_emission_train_model_min": float(results.loc[0, 'co2_emission_train_model_min']),
        "co2_emission_train_model_max": float(results.loc[0, 'co2_emission_train_model_max']),
        "response_time": float(results.loc[0, 'response_duration']),
        "tokens": int(results.loc[0, "response_token_length"])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)