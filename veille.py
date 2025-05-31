import streamlit as st
import feedparser
import requests
from datetime import datetime, timedelta
import os
from groq import Groq
import pandas as pd
from io import BytesIO
import concurrent.futures
from typing import List, Dict, Tuple
import time
import re
from html import unescape

# --- Configuration ---
PRODUCT_TYPES = ["Produits laitiers", "Viande", "Produits frais", "Produits de boulangerie", "Boissons", "Aliments transformés", "Autre"]
RISK_TYPES = ["Microbiologique", "Chimique", "Physique", "Allergène", "Fraude", "Autre"]
MARKETS = ["UE", "US", "Canada", "Royaume-Uni", "France", "International", "Autre"]

# Flux RSS optimisés avec gestion spéciale pour Health BE
FRENCH_EU_RSS_FEEDS = {
    "EFSA": "https://www.efsa.europa.eu/en/all/rss",
    "EU Food Safety": "https://food.ec.europa.eu/node/2/rss_en",
    "ANSES": "https://www.anses.fr/fr/flux-actualites.rss",
    "Health BE": "https://www.health.belgium.be/fr/rss/news.xml",
}

FOOD_SAFETY_KEYWORDS = [
    "rappel", "contamination", "allergène", "pathogène", "hygiène", "réglementation",
    "norme", "conformité", "audit", "inspection", "danger", "évaluation des risques",
    "maladie d'origine alimentaire", "traçabilité", "HACCP", "GFSI", "pesticide", "additif",
    "emballage", "étiquetage", "microbiologie", "toxicologie", "nouvel aliment", "fraude alimentaire"
]

# Configuration de pertinence
PERTINENCE_LEVELS = {
    "Très pertinent": {"score": 3, "color": "🟢", "min_threshold": 80},
    "Modérément pertinent": {"score": 2, "color": "🟡", "min_threshold": 60},
    "Peu pertinent": {"score": 1, "color": "🟠", "min_threshold": 40},
    "Non pertinent": {"score": 0, "color": "🔴", "min_threshold": 0}
}

# --- Classes et fonctions optimisées ---

def clean_html_content(html_content: str, preserve_basic_formatting: bool = True) -> str:
    """Nettoie le contenu HTML pour l'affichage et l'analyse."""
    if not html_content:
        return ""
    
    # Decode HTML entities
    cleaned = unescape(html_content)
    
    if preserve_basic_formatting:
        # Remplace les balises de paragraphe par des retours à la ligne
        cleaned = re.sub(r'</?p[^>]*>', '\n', cleaned)
        cleaned = re.sub(r'<br[^>]*/?>', '\n', cleaned)
        # Garde les listes avec des tirets
        cleaned = re.sub(r'<li[^>]*>', '- ', cleaned)
        cleaned = re.sub(r'</li>', '\n', cleaned)
        # Supprime toutes les autres balises HTML
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
    else:
        # Supprime toutes les balises HTML
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    # Nettoie les espaces multiples et retours à la ligne
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Double retours à la ligne max
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Espaces multiples
    cleaned = cleaned.strip()
    
    return cleaned

def extract_content_from_entry(entry) -> str:
    """Extrait le meilleur contenu disponible d'une entrée RSS."""
    content = ""
    
    # 1. Priorité : content:encoded (plus détaillé)
    if hasattr(entry, 'content') and entry.content:
        try:
            content = entry.content[0].value
        except (IndexError, AttributeError):
            pass
    
    # 2. Fallback : summary/description
    if not content and hasattr(entry, 'summary'):
        content = entry.summary
    
    # 3. Fallback final : title
    if not content:
        content = entry.title
    
    return content

class ArticleEvaluator:
    GROQ_MODEL = "llama3-8b-8192"
    MAX_TOKENS_TRANSLATE = 500
    MAX_TOKENS_EVALUATE = 600  # Augmenté pour permettre une évaluation plus détaillée

    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key) if groq_api_key else None
    
    def translate_to_french(self, text: str, content_type: str = "texte") -> Tuple[str, bool]:
        """
        Traduit un texte en français en utilisant Groq.
        Retourne un tuple: (texte_resultat, bool_tentative_traduction)
        bool_tentative_traduction est True si une requête API a été faite, False sinon.
        """
        translation_attempted = False
        if not self.client or not text or len(text.strip()) < 3:
            return text, translation_attempted
        
        # Détection plus robuste de l'anglais
        english_words = ['the', 'and', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must']
        text_lower = text.lower()
        
        # Compte les mots anglais
        word_count = len(text.split())
        english_count = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ' or text_lower.startswith(f'{word} ') or text_lower.endswith(f' {word}'))
        
        # Si moins de 20% de mots anglais détectés, probablement déjà en français
        if word_count > 0 and (english_count / word_count) < 0.2:
            return text, translation_attempted # Pas de tentative de traduction
        
        translation_attempted = True # Marquer comme tentative si on passe l'heuristique
        try:
            prompt = f"""
            Traduisez ce {content_type} scientifique en français professionnel.
            Gardez tous les termes techniques, noms d'espèces, codes de référence, et acronymes en version originale.
            
            IMPORTANT : Répondez UNIQUEMENT avec la traduction française, rien d'autre.
            
            Texte à traduire :
            {text}
            """
            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.GROQ_MODEL,
                temperature=0.0,  # Plus déterministe
                max_tokens=self.MAX_TOKENS_TRANSLATE,
            )
            
            translated = response.choices[0].message.content.strip()
            
            # Vérifie que la traduction n'est pas vide et différente de l'original
            if translated and len(translated) > 10 and translated != text:
                return translated, translation_attempted
            else:
                # La traduction a échoué ou n'est pas satisfaisante
                return text, translation_attempted
            
        except Exception as e:
            st.warning(f"Erreur de traduction pour {content_type}: {e}. Veuillez vérifier votre clé API Groq et votre connexion réseau.")
            return text, translation_attempted # Retourne l'original mais signale la tentative
        
    def evaluate_pertinence(self, article_title: str, article_summary: str, user_context: str) -> Tuple[str, str, int]:
        """Évalue la pertinence avec des critères TRÈS STRICTS sur la correspondance avec les types de produits."""
        if not self.client:
            return "Non pertinent", "Clé API manquante", 0

        prompt = f"""
        Vous êtes un expert en sécurité alimentaire européenne. Vous devez évaluer la pertinence de cet article pour un consultant/auditeur avec un profil TRÈS SPÉCIFIQUE.

        PROFIL UTILISATEUR DÉTAILLÉ:
        {user_context}

        ARTICLE À ÉVALUER:
        Titre: {article_title}
        Résumé: {article_summary}

        ⚠️ RÈGLES D'ÉVALUATION ULTRA-STRICTES ⚠️

        🔴 EXCLUSIONS AUTOMATIQUES (Score 0-20):
        - Alimentation animale / feed (sauf si profil inclut explicitement l'alimentation animale)
        - Cosmétiques, médicaments, dispositifs médicaux
        - Agriculture générale sans lien avec transformation alimentaire
        - Recherche fondamentale sans application pratique immédiate
        - Géographies non mentionnées dans le profil (ex: Asie si profil UE uniquement)
        - Types de produits NON listés dans le profil (ex: seafood si profil uniquement viande)
        - Articles génériques sur "food" sans spécification de produit

        🟠 PERTINENCE FAIBLE (Score 21-45):
        - Réglementation générale UE sans impact spécifique sur les produits du profil
        - Mentions tangentielles des types de produits sans focus principal
        - Contexte industrie alimentaire large mais pas les produits spécifiques
        - Nouvelles méthodologies analytiques générales

        🟡 PERTINENCE MODÉRÉE (Score 46-70):
        - Réglementation applicable aux produits du profil mais impact indirect
        - Types de risques mentionnés dans le profil mais sur d'autres produits
        - Évolutions générales des marchés spécifiés
        - Études sur des dangers pertinents mais produits différents

        🟢 TRÈS PERTINENT (Score 71-100):
        - CORRESPONDANCE EXACTE : Type de produit + Type de risque + Marché du profil
        - Nouvelles réglementations DIRECTEMENT applicables aux produits du profil
        - Rappels/alertes sur les types de produits spécifiques du profil
        - Nouveaux dangers sur les produits et marchés du profil
        - Nouvelles méthodes d'analyse pour les produits du profil

        EXEMPLES CONCRETS DE SCORING:
        
        Si profil = "Produits laitiers, Microbiologique, UE":
        - Article sur Listeria dans fromages UE → Score 90-95 (PARFAIT)
        - Article sur STEC dans lait cru France → Score 85-90 (EXCELLENT)
        - Article sur pesticides dans légumes UE → Score 15-25 (MAUVAIS: type produit)
        - Article sur Salmonella dans produits laitiers US → Score 35-45 (MOYEN: mauvais marché)
        - Article réglementation générale UE → Score 40-50 (FAIBLE: trop général)
        - Article sur feed contamination → Score 0-10 (EXCLUSION)

        Si profil = "Viande, Chimique, France":
        - Article sur résidus antibiotiques viande France → Score 90-100 (PARFAIT)
        - Article sur allergènes produits laitiers France → Score 20-30 (MAUVAIS: produit + risque)
        - Article sur hormones viande UE → Score 65-75 (BON: produit/risque, géo proche)
        - Article sur Campylobacter volaille France → Score 45-55 (MOYEN: bon produit/géo, mauvais risque)

        Si profil = "Autre" (tous produits):
        - Soyez moins strict sur les types de produits mais maintenez les autres critères

        ATTENTION MAXIMALE:
        ❌ Soyez IMPITOYABLE sur la correspondance exacte avec les types de produits
        ❌ Un article sur "légumes" n'est PAS pertinent pour un profil "produits laitiers"
        ❌ Un article "alimentation animale" n'est PAS pertinent pour "produits pour humains"
        ❌ Un article "cosmétiques" n'est JAMAIS pertinent pour food safety
        ❌ Géographie : USA/Canada uniquement pertinents si mentionnés dans profil
        ❌ Articles génériques sur "sécurité alimentaire" sans spécification = score faible

        PROCESSUS D'ÉVALUATION OBLIGATOIRE:
        1. Identifiez le type de produit principal de l'article
        2. Vérifiez s'il correspond EXACTEMENT à un type du profil
        3. Identifiez le type de risque de l'article
        4. Vérifiez la géographie/marché
        5. Si une des correspondances majeures échoue → Score maximum 45
        6. Soyez particulièrement strict si le profil ne contient PAS "Autre"

        Répondez EXACTEMENT dans ce format:
        Pertinence: [Très pertinent/Modérément pertinent/Peu pertinent/Non pertinent]
        Score: [nombre entre 0 et 100]
        Résumé: [En 2-3 phrases, expliquez la correspondance EXACTE ou les éléments qui ne correspondent PAS au profil. Mentionnez explicitement les types de produits, risques et marchés.]
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.GROQ_MODEL,
                temperature=0.0,  # Plus déterministe pour l'évaluation
                max_tokens=self.MAX_TOKENS_EVALUATE,
            )
            
            content = response.choices[0].message.content
            return self._parse_evaluation(content)
            
        except Exception as e:
            st.error(f"Erreur API Groq: {e}. Veuillez vérifier votre clé API Groq et votre connexion réseau.")
            return "Non pertinent", "Erreur d'évaluation", 0
    
    def _parse_evaluation(self, response: str) -> Tuple[str, str, int]:
        """Parse la réponse de l'API pour extraire pertinence, score et résumé."""
        # Valeurs par défaut sécurisées
        pertinence_level = "Non pertinent"
        summary = "Impossible d'évaluer"
        score = 0
        
        if not response:
            return pertinence_level, summary, score
        
        try:
            lines = response.split('\n')
            for line in lines:
                line_stripped = line.strip() # For whitespace
                if line_stripped.lower().startswith("pertinence:"):
                    level_text = line_stripped[len("Pertinence:"):].strip()
                    # Validation des niveaux acceptés
                    valid_levels = ["Très pertinent", "Modérément pertinent", "Peu pertinent", "Non pertinent"]
                    if level_text in valid_levels: # Preserves original casing if valid
                        pertinence_level = level_text
                    # Handle cases where LLM might output "Pertinence: Très Pertinent" (title case)
                    # by checking title case version if direct match fails.
                    elif level_text.title() in valid_levels:
                         pertinence_level = level_text.title()

                elif line_stripped.lower().startswith("score:"):
                    score_text = line_stripped[len("Score:"):].strip()
                    try:
                        # Extraction du nombre même s'il y a du texte autour
                        score_match = re.search(r'\d+', score_text)
                        if score_match:
                            score = int(score_match.group())
                            score = max(0, min(100, score))  # Clamp entre 0 et 100
                    except (ValueError, AttributeError):
                        score = 0 # Default if parsing fails
                elif line_stripped.lower().startswith("résumé:"): # French 'Résumé'
                    summary_text = line_stripped[len("Résumé:"):].strip()
                    if summary_text: # Ensure not empty
                        summary = summary_text[:400]  # Limite la taille augmentée
        
        except Exception as e:
            st.warning(f"Erreur lors du parsing de l'évaluation : {e}")
        
        return pertinence_level, summary, score

def apply_additional_filtering(evaluated_articles: List[Dict], user_context: str) -> List[Dict]:
    """
    Applique un filtrage supplémentaire basé sur des mots-clés pour s'assurer 
    que les articles correspondent vraiment aux types de produits du profil.
    """
    
    # Extraction des types de produits du contexte utilisateur
    context_lower = user_context.lower()
    
    # Définition des mots-clés par type de produit (français et anglais)
    product_keywords = {
        "produits laitiers": [
            "lait", "fromage", "yaourt", "yogourt", "beurre", "crème", "lactose", "lactosérum", "whey", "caséine",
            "dairy", "cheese", "milk", "yogurt", "yoghurt", "butter", "cream", "lactose", "casein"
        ],
        "viande": [
            "viande", "porc", "bœuf", "boeuf", "volaille", "poulet", "porc", "agneau", "veau", "jambon", "saucisse",
            "meat", "beef", "pork", "chicken", "poultry", "lamb", "veal", "ham", "sausage", "bacon"
        ],
        "produits frais": [
            "légumes", "fruits", "salade", "épinards", "tomates", "pommes", "bananes", "fraises", "herbes",
            "fresh", "vegetable", "fruit", "produce", "frais", "salad", "spinach", "tomato", "apple", "banana", "strawberry"
        ],
        "produits de boulangerie": [
            "pain", "boulangerie", "pâtisserie", "blé", "farine", "gluten", "céréales", "biscuit", "gâteau",
            "bakery", "bread", "wheat", "flour", "gluten", "cereal", "cookie", "cake", "pastry"
        ],
        "boissons": [
            "boisson", "jus", "eau", "soda", "thé", "café", "vin", "bière", "alcool",
            "beverage", "drink", "juice", "water", "soft drink", "tea", "coffee", "wine", "beer", "alcohol"
        ],
        "aliments transformés": [
            "transformé", "préparé", "conserve", "surgelé", "plat préparé", "sauce", "condiment",
            "processed", "prepared", "manufactured", "canned", "frozen", "ready meal", "sauce", "condiment"
        ],
        "autre": []  # Le type "autre" accepte tout
    }
    
    # Mots-clés d'exclusion (alimentation animale, etc.)
    exclusion_keywords = [
        "feed", "animal feed", "alimentation animale", "aliment pour animaux", "aliments pour animaux",
        "pet food", "nourriture pour animaux", "food for animals", "animal nutrition",
        "cosmetic", "cosmétique", "cosmetics", "cosmétiques",
        "medical device", "dispositif médical", "pharmaceutical", "pharmaceutique", "drug",
        "fertilizer", "fertilisant", "pesticide application", "agricultural chemicals"
    ]
    
    # Mots-clés de recherche fondamentale (moins prioritaires)
    research_keywords = [
        "in vitro", "in vivo", "laboratory study", "étude de laboratoire", "recherche fondamentale",
        "basic research", "theoretical", "théorique", "model", "modèle", "simulation"
    ]
    
    # Identifier les types de produits dans le profil utilisateur
    user_product_types = []
    for product_type in product_keywords.keys():
        if product_type in context_lower:
            user_product_types.append(product_type)
    
    # Si "autre" est dans le profil, on est moins restrictif
    if "autre" in user_product_types:
        less_restrictive = True
    else:
        less_restrictive = False
    
    filtered_articles = []
    
    for article in evaluated_articles:
        title_lower = article['Titre'].lower()
        summary_lower = article['Résumé'].lower()
        article_text = f"{title_lower} {summary_lower}"
        
        # Calcul de pénalités
        penalty = 0
        reasons = []
        
        # Vérification des mots-clés d'exclusion STRICTE
        exclusion_found = []
        for keyword in exclusion_keywords:
            if keyword in article_text:
                exclusion_found.append(keyword)
        
        if exclusion_found:
            penalty += 60  # Pénalité très lourde
            reasons.append(f"Exclusion détectée: {', '.join(exclusion_found[:2])}")
        
        # Vérification de la correspondance avec les types de produits
        if not less_restrictive and user_product_types:
            product_match_found = False
            matched_products = []
            
            for user_product_type in user_product_types:
                if user_product_type in product_keywords:
                    keywords = product_keywords[user_product_type]
                    for keyword in keywords:
                        if keyword in article_text:
                            product_match_found = True
                            matched_products.append(keyword)
                            break
                    if product_match_found:
                        break
            
            # Si aucune correspondance trouvée, pénalité importante
            if not product_match_found:
                penalty += 40
                reasons.append("Aucune correspondance type de produit")
            elif matched_products:
                reasons.append(f"Produit détecté: {matched_products[0]}")
        
        # Vérification recherche fondamentale
        research_found = any(keyword in article_text for keyword in research_keywords)
        if research_found:
            penalty += 15
            reasons.append("Recherche fondamentale")
        
        # Application des pénalités
        original_score = article['Score']
        article['Score'] = max(0, original_score - penalty)
        
        # Mise à jour de l'évaluation avec les raisons
        if reasons:
            article['Évaluation de la Pertinence'] += f" [FILTRAGE: {'; '.join(reasons)}]"
        
        # Log des modifications importantes
        if penalty > 30:
            article['Évaluation de la Pertinence'] += f" [SCORE: {original_score}→{article['Score']}]"
        
        # Seuil d'exclusion après filtrage
        if article['Score'] >= 25:  # Seuil minimum après filtrage
            filtered_articles.append(article)
    
    return filtered_articles

@st.cache_data(ttl=1800, show_spinner="Récupération RSS...")
def fetch_rss_feed(url: str) -> List[Dict]:
    """Récupère les entrées RSS avec gestion d'erreur améliorée."""
    try:
        # Headers pour éviter les blocages
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Tentative avec requests d'abord pour les URLs problématiques
        if 'health.belgium.be' in url:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
            except Exception as e:
                st.warning(f"Erreur requests pour {url}, tentative feedparser direct: {e}")
                feed = feedparser.parse(url)
        else:
            feed = feedparser.parse(url)
        
        if feed.bozo and feed.bozo_exception:
            st.warning(f"Problème de parsing pour {url}: {feed.bozo_exception}")
        
        return feed.entries
        
    except Exception as e:
        st.error(f"Erreur RSS {url}: {e}")
        return []

def fetch_all_articles(feeds: Dict[str, str], start_date: datetime.date, end_date: datetime.date, clean_html: bool = True) -> List[Dict]:
    """Récupère tous les articles en parallèle avec extraction optimisée du contenu."""
    all_articles = []
    failed_feeds = []
    
    # Récupération de l'option de formatage depuis session_state ou valeur par défaut
    preserve_formatting = st.session_state.get('preserve_formatting', True)
    
    with st.spinner("Récupération des flux RSS..."):
        progress_bar = st.progress(0)
        
        for idx, (source_name, url) in enumerate(feeds.items()):
            try:
                with st.spinner(f"Récupération de {source_name}..."):
                    entries = fetch_rss_feed(url)
                    
                    if not entries:
                        st.warning(f"Aucune entrée trouvée pour {source_name}")
                        if source_name not in failed_feeds: # Avoid duplicates
                            failed_feeds.append(source_name)
                        continue
                
                for entry in entries:
                    try:
                        published_date = None
                        
                        # Gestion améliorée des dates
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_date = datetime(*entry.published_parsed[:6]).date()
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            published_date = datetime(*entry.updated_parsed[:6]).date()
                        elif hasattr(entry, 'published'):
                            try:
                                # Essayer différents formats de date
                                for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%a, %d %b %Y %H:%M:%S %z', '%Y-%m-%d']:
                                    try:
                                        published_date = datetime.strptime(entry.published, fmt).date()
                                        break
                                    except ValueError:
                                        continue
                            except Exception as e_parse_str_date:
                                # If parsing fails, default to now, and optionally log or inform user
                                # For now, just using a default. A debug message could be added.
                                # st.sidebar.caption(f"Debug: Date parse error for '{entry.get('title', 'Unknown title')}', used current date. Error: {e_parse_str_date}")
                                published_date = datetime.now().date()
                        
                        # Si pas de date trouvée, utiliser la date actuelle
                        if not published_date:
                            published_date = datetime.now().date()
                        
                        if start_date <= published_date <= end_date:
                            # Extraction optimisée du contenu
                            raw_content = extract_content_from_entry(entry)
                            
                            # Validation du contenu
                            if not raw_content or len(raw_content.strip()) < 10:
                                raw_content = getattr(entry, 'title', 'Contenu non disponible')
                            
                            # Nettoyage HTML optionnel
                            if clean_html:
                                summary = clean_html_content(raw_content, preserve_basic_formatting=preserve_formatting)
                            else:
                                summary = raw_content
                            
                            # Validation finale
                            if not summary or len(summary.strip()) < 5:
                                summary = "Résumé non disponible"
                            
                            all_articles.append({
                                "source": source_name,
                                "title": getattr(entry, 'title', 'Titre non disponible'),
                                "summary": summary,
                                "raw_content": raw_content,
                                "link": getattr(entry, 'link', '#'),
                                "published": published_date
                            })
                            
                    except Exception as e:
                        st.warning(f"Erreur lors du traitement d'un article de {source_name}: {e}")
                        continue
                        
                progress_bar.progress((idx + 1) / len(feeds))
                
            except Exception as e:
                st.error(f"Erreur lors de la récupération de {source_name}: {e}")
                if source_name not in failed_feeds: # Avoid duplicates
                    failed_feeds.append(source_name)
                continue
    
    st.success(f"✅ {len(all_articles)} articles récupérés au total")

    if failed_feeds:
        unique_failed_feeds = sorted(list(set(failed_feeds)))
        st.warning(f"⚠️ Problèmes rencontrés avec les flux suivants : {', '.join(unique_failed_feeds)}. Certains articles de ces sources pourraient manquer.")

    return all_articles

# --- Interface Streamlit optimisée ---

st.set_page_config(
    page_title="Veille Réglementaire Food Safety", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Veille Réglementaire - Sécurité Alimentaire")
st.markdown("*Application optimisée pour consultants et auditeurs food safety*")
st.markdown("✨ **Nouveau** : Évaluation ultra-stricte de la pertinence + Filtrage avancé par type de produit")

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Seuil de pertinence
    min_pertinence_score = st.slider(
        "Score minimum de pertinence", 
        min_value=0, 
        max_value=100, 
        value=60,
        help="Articles avec un score inférieur seront filtrés"
    )
    
    # Nombre max d'articles
    max_articles = st.selectbox(
        "Nombre maximum d'articles à évaluer",
        [10, 20, 50, 100],
        index=1
    )
    
    # Options de filtrage strict
    st.subheader("🎯 Filtrage Strict")
    
    enable_strict_filtering = st.checkbox(
        "Activer le filtrage strict par type de produit",
        value=True,
        help="Applique des pénalités automatiques pour les articles non-correspondants"
    )
    
    exclude_research = st.checkbox(
        "Exclure la recherche fondamentale",
        value=True,
        help="Pénalise les articles de recherche pure sans application pratique"
    )
    
    # Options de parsing
    st.subheader("🔧 Options de parsing")
    
    st.session_state['clean_html'] = st.checkbox(
        "Nettoyer le HTML des résumés",
        value=True,
        help="Convertit le HTML en texte lisible (recommandé pour Health BE)"
    )
    
    st.session_state['preserve_formatting'] = st.checkbox(
        "Préserver le formatage de base",
        value=True,
        help="Garde les paragraphes et listes lors du nettoyage HTML"
    )
    
    # Option de traduction
    st.session_state['translate_to_french'] = st.checkbox(
        "🇫🇷 Traduire le contenu en français",
        value=False,
        help="Traduit automatiquement les titres et résumés en français (utilise l'API Groq)"
    )
    
    # Information sur l'extraction de contenu
    with st.expander("ℹ️ Filtrage & Extraction"):
        st.markdown("""
        **🎯 Filtrage Ultra-Strict:**
        - Exclusion automatique : alimentation animale, cosmétiques
        - Correspondance exacte types de produits requis
        - Pénalités pour recherche fondamentale
        
        **📊 Extraction de contenu:**
        1. `content:encoded` (plus détaillé, notamment Health BE)
        2. `summary/description` (fallback standard)
        3. `title` (fallback minimal)
        
        **🇫🇷 Traduction** : Les articles EFSA/EU (en anglais) peuvent être traduits automatiquement
        """)
    
# --- Profil utilisateur (simplifié) ---
with st.expander("🏢 Profil d'Activité", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        selected_product_types = st.multiselect(
            "Types de Produits",
            PRODUCT_TYPES,
            default=["Autre"],
            help="⚠️ Sélection très importante pour le filtrage strict!"
        )
        
        selected_risk_types = st.multiselect(
            "Types de Risques",
            RISK_TYPES,
            default=["Microbiologique", "Chimique"]
        )
    
    with col2:
        selected_markets = st.multiselect(
            "Marchés Cibles",
            MARKETS,
            default=["UE", "France"]
        )
        
        main_concerns = st.text_area(
            "Préoccupations Spécifiques",
            placeholder="Ex: PFAS, Listeria, nouvelles réglementations EU...",
            height=100
        )

# Création du contexte utilisateur
user_context = f"""
Types de Produits: {', '.join(selected_product_types)}
Types de Risques: {', '.join(selected_risk_types)}
Marchés: {', '.join(selected_markets)}
Préoccupations: {main_concerns}
Profil: Consultant/Auditeur en sécurité alimentaire européenne
"""

# --- Configuration période et API ---
col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    start_date = st.date_input("📅 Date de Début", datetime.now() - timedelta(days=7))

with col2:
    end_date = st.date_input("📅 Date de Fin", datetime.now())

with col3:
    groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = st.text_input("🔑 Clé API Groq", type="password", 
                                   help="Nécessaire pour l'évaluation de pertinence")

# --- Lancement de la veille ---
if st.button("🚀 Lancer la Veille", type="primary", use_container_width=True, disabled=not groq_api_key):
    if not groq_api_key:
        st.error("Clé API Groq requise pour l'évaluation de pertinence")
        st.stop()
    
    if start_date > end_date:
        st.error("La date de fin doit être postérieure à la date de début")
        st.stop()
    
    # Récupération des articles
    all_articles = fetch_all_articles(FRENCH_EU_RSS_FEEDS, start_date, end_date, st.session_state.get('clean_html', True))
    
    if not all_articles:
        st.warning("Aucun article trouvé dans la période spécifiée")
        st.stop()
    
    # Limitation du nombre d'articles à évaluer
    all_articles = all_articles[:max_articles]
    
    # Message informatif sur la traduction
    translate_to_french = st.session_state.get('translate_to_french', False)
    if translate_to_french:
        st.info(f"🇫🇷 **Mode traduction activé** : Les titres et résumés en anglais seront traduits automatiquement")
        st.info("➡️ Vérifiez la sidebar pour les détails de traduction en temps réel")
    
    st.info(f"🔍 Évaluation de {len(all_articles)} articles avec critères ultra-stricts...")
    
    # Évaluation des articles avec traduction optionnelle
    evaluator = ArticleEvaluator(groq_api_key)
    evaluated_articles = []
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Option de traduction et conteneur de debug
    translate_to_french = st.session_state.get('translate_to_french', False)
    translation_debug = None
    
    if translate_to_french:
        # Créer un conteneur de debug dans la sidebar
        with st.sidebar:
            st.markdown("**🔍 Debug Traduction en cours:**")
            translation_debug = st.empty()
    
    for idx, article in enumerate(all_articles):
        progress_text.text(f"Évaluation {idx+1}/{len(all_articles)}: {article['source']}")
        
        # Traduction optionnelle AVANT évaluation
        current_title = article['title']
        current_summary = article['summary']
        translation_performed_for_article = False # Pour marquer si au moins un des champs a été réellement changé
        
        if translate_to_french:
            progress_text.text(f"Traduction {idx+1}/{len(all_articles)}: {article['source']}")
            
            original_title = article['title']
            original_summary = article['summary']

            translated_title_text, title_translation_attempted = evaluator.translate_to_french(original_title, "titre")
            summary_translation_text, summary_translation_attempted = evaluator.translate_to_french(original_summary, "résumé")

            title_changed = (translated_title_text != original_title)
            summary_changed = (summary_translation_text != original_summary)

            current_title = translated_title_text
            current_summary = summary_translation_text
            
            if title_changed or summary_changed:
                translation_performed_for_article = True

            # Debug info dans la sidebar
            if translation_debug is not None:
                # Titre
                if not title_translation_attempted:
                    translation_debug.info(f"ℹ️ Titre Art.{idx+1} ({article['source']}): Non traduit (déjà FR)")
                elif title_changed:
                    translation_debug.success(f"✅ Titre Art.{idx+1} ({article['source']}): Traduit")
                else: # Tentative mais pas de changement ou échec qualité
                    translation_debug.warning(f"⚠️ Titre Art.{idx+1} ({article['source']}): Original conservé (trad. insatisfaisante/échec)")

                # Résumé
                if not summary_translation_attempted:
                    translation_debug.info(f"ℹ️ Résumé Art.{idx+1} ({article['source']}): Non traduit (déjà FR)")
                elif summary_changed:
                    translation_debug.success(f"✅ Résumé Art.{idx+1} ({article['source']}): Traduit")
                else: # Tentative mais pas de changement ou échec qualité
                    translation_debug.warning(f"⚠️ Résumé Art.{idx+1} ({article['source']}): Original conservé (trad. insatisfaisante/échec)")

        progress_text.text(f"Évaluation {idx+1}/{len(all_articles)}: {article['source']}")
        
        # Évaluation avec le contenu (possiblement traduit)
        pertinence_level, evaluation_summary, score = evaluator.evaluate_pertinence(
            current_title,
            current_summary,
            user_context
        )
        
        # Filtrage basé sur le score minimum
        if score >= min_pertinence_score:
            evaluated_articles.append({
                "Source": article['source'],
                "Titre": current_title,  # Titre possiblement traduit
                "Résumé": current_summary,  # Résumé possiblement traduit
                "Titre Original": article['title'],  # Garde l'original pour référence
                "Résumé Original": article['summary'],  # Garde l'original pour référence
                "Lien": article['link'],
                "Date de Publication": article['published'].strftime('%Y-%m-%d'),
                "Niveau de Pertinence": pertinence_level,
                "Score": score,
                "Évaluation de la Pertinence": evaluation_summary,
                "raw_content": article.get('raw_content', ''),
                "Traduit": translation_performed_for_article # Utilise le nouveau flag
            })
        
        progress_bar.progress((idx + 1) / len(all_articles))
    
    progress_text.empty()
    progress_bar.empty()
    
    # Nettoyage du debug de traduction
    if translate_to_french and translation_debug is not None:
        translation_debug.empty()
    
    # NOUVEAU: Filtrage supplémentaire basé sur les mots-clés
    if evaluated_articles and enable_strict_filtering:
        articles_before_filtering = len(evaluated_articles)
        st.info("🎯 Application du filtrage strict basé sur les types de produits...")
        evaluated_articles = apply_additional_filtering(evaluated_articles, user_context)
        
        # Re-filtrage basé sur le score minimum après ajustements
        evaluated_articles = [a for a in evaluated_articles if a['Score'] >= min_pertinence_score]
        
        articles_after_filtering = len(evaluated_articles)
        if articles_before_filtering != articles_after_filtering:
            st.warning(f"⚠️ Filtrage strict : {articles_before_filtering - articles_after_filtering} articles supplémentaires filtrés")
    
    # Affichage des résultats
    if evaluated_articles:
        # Tri par score décroissant
        evaluated_articles.sort(key=lambda x: x['Score'], reverse=True)
        
        st.success(f"✅ {len(evaluated_articles)} articles pertinents trouvés après filtrage strict (score ≥ {min_pertinence_score})")
        
        # Métriques améliorées
        col1, col2, col3, col4, col5 = st.columns(5)
        
        score_stats = [a['Score'] for a in evaluated_articles]
        with col1:
            st.metric("Score Moyen", f"{sum(score_stats)/len(score_stats):.1f}/100")
        with col2:
            very_pertinent = len([a for a in evaluated_articles if a['Score'] >= 80])
            st.metric("Très Pertinents", very_pertinent)
        with col3:
            moderate_pertinent = len([a for a in evaluated_articles if 60 <= a['Score'] < 80])
            st.metric("Modérément Pertinents", moderate_pertinent)
        with col4:
            st.metric("Score Max", f"{max(score_stats)}/100")
        with col5:
            translated_count = len([a for a in evaluated_articles if a.get('Traduit', False)])
            st.metric("Traduits", f"{translated_count}/{len(evaluated_articles)}")
        
        # AFFICHAGE SIMPLE - AUCUN data_editor !
        st.subheader("📊 Articles Sélectionnés")
        
        # 1. Vue d'ensemble avec tableau simple
        st.markdown("### 📋 Vue d'ensemble")
        
        # Préparation simple des données
        simple_data = []
        for article in evaluated_articles:
            score_visual = f"⭐ {article['Score']}/100 " + ("🟢" if article['Score'] >= 80 else "🟡" if article['Score'] >= 60 else "🟠")
            lien_status = "🔗 Disponible" if article['Lien'] != '#' else "❌ Indisponible"
            
            # Indicateur de traduction
            title_display = article['Titre'][:80] + ("..." if len(article['Titre']) > 80 else "")
            if article.get('Traduit', False):
                title_display = f"🇫🇷 {title_display}"
            
            simple_data.append({
                'Source': article['Source'],
                'Titre': title_display,
                'Score': score_visual,
                'Date': article['Date de Publication'],
                'Lien': lien_status
            })
        
        # Affichage tableau simple
        simple_df = pd.DataFrame(simple_data)
        st.dataframe(simple_df, use_container_width=True)
        
        # 2. Sélection pour export
        st.markdown("### 📥 Sélection pour Export")
        
        options_for_export = []
        for idx, article in enumerate(evaluated_articles):
            # Ajoute un indicateur de traduction dans le label
            translation_indicator = " 🇫🇷" if article.get('Traduit', False) else ""
            label = f"⭐{article['Score']} - {article['Source']} - {article['Titre'][:50]}...{translation_indicator}"
            options_for_export.append((idx, label))
        
        selected_indices = st.multiselect(
            "Sélectionnez les articles à exporter :",
            options=[opt[0] for opt in options_for_export],
            format_func=lambda x: next(opt[1] for opt in options_for_export if opt[0] == x),
            default=list(range(min(5, len(options_for_export))))
        )
        
        # 3. Affichage détaillé
        if selected_indices:
            st.markdown(f"### 📖 Détails des {len(selected_indices)} articles sélectionnés")
            
            for idx in selected_indices:
                if idx < len(evaluated_articles):
                    article = evaluated_articles[idx]
                    
                    with st.expander(f"⭐ {article['Score']}/100 - {article['Source']} - {article['Titre'][:60]}..."):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Affichage du titre avec indicateur de traduction
                            if article.get('Traduit', False):
                                st.markdown(f"**📰 Titre (🇫🇷 traduit) :** {article['Titre']}")
                                
                                # Utilisation d'un toggle button au lieu d'un expander imbriqué
                                show_original_title = st.checkbox("👁️ Voir titre original", key=f"title_orig_{idx}")
                                if show_original_title:
                                    st.info(f"**🔤 Titre original :** {article.get('Titre Original', 'N/A')}")
                            else:
                                st.markdown(f"**📰 Titre :** {article['Titre']}")
                            
                            # Affichage du résumé avec indicateur de traduction
                            if article.get('Traduit', False):
                                st.markdown(f"**📝 Résumé (🇫🇷 traduit) :**")
                                st.write(article['Résumé'])
                                
                                # Utilisation d'un toggle button au lieu d'un expander imbriqué
                                show_original_summary = st.checkbox("👁️ Voir résumé original", key=f"summary_orig_{idx}")
                                if show_original_summary:
                                    st.info(f"**📝 Résumé original :**")
                                    st.write(article.get('Résumé Original', 'N/A'))
                            else:
                                st.markdown(f"**📝 Résumé :**")
                                st.write(article['Résumé'])
                            
                            st.markdown(f"**🔍 Évaluation :**")
                            st.write(article['Évaluation de la Pertinence'])
                        
                        with col2:
                            st.markdown(f"**📡 Source :** {article['Source']}")
                            st.markdown(f"**⭐ Score :** {article['Score']}/100")
                            st.markdown(f"**📅 Date :** {article['Date de Publication']}")
                            
                            if article['Lien'] != '#':
                                st.markdown(f"**[🔗 Ouvrir l'article]({article['Lien']})**")
                            else:
                                st.markdown("**❌ Lien indisponible**")
        else:
            st.info("Sélectionnez des articles ci-dessus pour voir les détails")
        
        # 4. Debug avancé
        if st.sidebar.checkbox("🔬 Mode Debug - Afficher contenu brut"):
            st.markdown("### 🔬 Comparaison Contenu Nettoyé vs Brut")
            
            debug_article_idx = st.selectbox(
                "Sélectionnez un article pour le debug :",
                range(len(evaluated_articles)),
                format_func=lambda x: f"{evaluated_articles[x]['Source']} - {evaluated_articles[x]['Titre'][:50]}..."
            )
            
            if debug_article_idx < len(evaluated_articles):
                article = evaluated_articles[debug_article_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📝 Contenu Nettoyé:**")
                    st.text_area("", article['Résumé'], height=200, key="clean_content_debug")
                
                with col2:
                    st.markdown("**🔧 Contenu Brut (HTML):**")
                    raw_content = article.get('raw_content', 'Non disponible')
                    st.text_area("", raw_content, height=200, key="raw_content_debug")
        
        # 5. Statistiques de filtrage
        if st.sidebar.checkbox("📊 Statistiques de filtrage"):
            st.markdown("### 📊 Analyse du Filtrage")
            
            # Analyse des évaluations contenant des alertes de filtrage
            filtered_evaluations = [a for a in evaluated_articles if "[FILTRAGE:" in a['Évaluation de la Pertinence']]
            
            if filtered_evaluations:
                st.info(f"🎯 {len(filtered_evaluations)} articles ont subi un filtrage supplémentaire")
                
                # Extraction des raisons de filtrage
                filter_reasons = {}
                for article in filtered_evaluations:
                    eval_text = article['Évaluation de la Pertinence']
                    if "[FILTRAGE:" in eval_text:
                        reason_part = eval_text.split("[FILTRAGE:")[1].split("]")[0]
                        reasons = reason_part.split(";")
                        for reason in reasons:
                            reason = reason.strip()
                            if reason:
                                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                
                if filter_reasons:
                    st.markdown("**Raisons de filtrage :**")
                    for reason, count in sorted(filter_reasons.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- {reason}: {count} articles")
            else:
                st.success("✅ Aucun article n'a nécessité de filtrage supplémentaire")
        
        # Téléchargement
        if len(evaluated_articles) > 0:
            st.subheader("💾 Téléchargement")
            
            col1, col2, col3 = st.columns(3)
            
            # Préparation données export (nettoie les colonnes techniques)
            if selected_indices:
                selected_articles_data = [evaluated_articles[i] for i in selected_indices if i < len(evaluated_articles)]
                # Nettoyage pour export
                clean_articles = []
                for article in selected_articles_data:
                    clean_article = {
                        'Source': article['Source'],
                        'Titre': article['Titre'],
                        'Résumé': article['Résumé'],
                        'Lien': article['Lien'],
                        'Date de Publication': article['Date de Publication'],
                        'Niveau de Pertinence': article['Niveau de Pertinence'],
                        'Score': article['Score'],
                        'Évaluation de la Pertinence': article['Évaluation de la Pertinence']
                    }
                    # Ajoute les versions originales si traduit
                    if article.get('Traduit', False):
                        clean_article['Titre Original'] = article.get('Titre Original', '')
                        clean_article['Résumé Original'] = article.get('Résumé Original', '')
                        clean_article['Traduit'] = 'Oui'
                    else:
                        clean_article['Traduit'] = 'Non'
                    clean_articles.append(clean_article)
                
                export_df = pd.DataFrame(clean_articles)
                st.success(f"📌 {len(selected_articles_data)} articles sélectionnés pour l'export")
            else:
                # Même nettoyage pour tous les articles
                clean_articles = []
                for article in evaluated_articles:
                    clean_article = {
                        'Source': article['Source'],
                        'Titre': article['Titre'],
                        'Résumé': article['Résumé'],
                        'Lien': article['Lien'],
                        'Date de Publication': article['Date de Publication'],
                        'Niveau de Pertinence': article['Niveau de Pertinence'],
                        'Score': article['Score'],
                        'Évaluation de la Pertinence': article['Évaluation de la Pertinence']
                    }
                    if article.get('Traduit', False):
                        clean_article['Titre Original'] = article.get('Titre Original', '')
                        clean_article['Résumé Original'] = article.get('Résumé Original', '')
                        clean_article['Traduit'] = 'Oui'
                    else:
                        clean_article['Traduit'] = 'Non'
                    clean_articles.append(clean_article)
                
                export_df = pd.DataFrame(clean_articles)
                st.info("📌 Aucune sélection - tous les articles seront exportés")
            
            with col1:
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv_data,
                    file_name=f"veille_food_safety_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, sheet_name='Veille Food Safety', index=False)
                
                excel_buffer.seek(0)
                st.download_button(
                    label="📥 Télécharger Excel",
                    data=excel_buffer,
                    file_name=f"veille_food_safety_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col3:
                if selected_indices:
                    scores = [evaluated_articles[i]['Score'] for i in selected_indices if i < len(evaluated_articles)]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        st.metric("Score Moyen Sélection", f"{avg_score:.1f}/100")
                else:
                    st.info("Aucune sélection active")
    
    else:
        st.warning(f"❌ Aucun article avec un score ≥ {min_pertinence_score} trouvé après filtrage strict. Essayez de :")
        st.markdown("""
        - Réduire le seuil de pertinence
        - Désactiver le filtrage strict
        - Élargir les types de produits (ajouter "Autre")
        - Vérifier la correspondance avec votre profil
        """)

# Footer amélioré
st.markdown("---")
st.markdown("*Développé pour les professionnels de la sécurité alimentaire européenne*")
st.markdown("🎯 **Version 2.0** : Filtrage ultra-strict + Traduction automatique + Export optimisé")
