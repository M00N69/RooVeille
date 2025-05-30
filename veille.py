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
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key) if groq_api_key else None
        
    def evaluate_pertinence(self, article_title: str, article_summary: str, user_context: str) -> Tuple[str, str, int]:
        """Évalue la pertinence avec un score numérique pour un meilleur filtrage."""
        if not self.client:
            return "Non pertinent", "Clé API manquante", 0

        prompt = f"""
        Vous êtes un expert en sécurité alimentaire. Évaluez la pertinence de cet article pour un professionnel avec le profil suivant.

        PROFIL UTILISATEUR:
        {user_context}

        ARTICLE À ÉVALUER:
        Titre: {article_title}
        Résumé: {article_summary}

        Critères d'évaluation:
        - Très pertinent (80-100): Impact direct sur l'activité, réglementation applicable, risque majeur
        - Modérément pertinent (60-79): Intérêt professionnel, veille concurrentielle, évolution réglementaire
        - Peu pertinent (40-59): Information générale, contexte industrie
        - Non pertinent (0-39): Hors sujet, pas d'impact

        Répondez EXACTEMENT dans ce format:
        Pertinence: [Très pertinent/Modérément pertinent/Peu pertinent/Non pertinent]
        Score: [nombre entre 0 et 100]
        Résumé: [En 1-2 phrases, pourquoi cet article est pertinent ou non pour ce profil]
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.1,
                max_tokens=300,
            )
            
            content = response.choices[0].message.content
            return self._parse_evaluation(content)
            
        except Exception as e:
            st.error(f"Erreur API Groq: {e}")
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
                line = line.strip()
                if line.startswith("Pertinence:"):
                    level_text = line.replace("Pertinence:", "").strip()
                    # Validation des niveaux acceptés
                    valid_levels = ["Très pertinent", "Modérément pertinent", "Peu pertinent", "Non pertinent"]
                    if level_text in valid_levels:
                        pertinence_level = level_text
                elif line.startswith("Score:"):
                    try:
                        score_text = line.replace("Score:", "").strip()
                        # Extraction du nombre même s'il y a du texte autour
                        score_match = re.search(r'\d+', score_text)
                        if score_match:
                            score = int(score_match.group())
                            score = max(0, min(100, score))  # Clamp entre 0 et 100
                    except (ValueError, AttributeError):
                        score = 0
                elif line.startswith("Résumé:"):
                    summary_text = line.replace("Résumé:", "").strip()
                    if summary_text:
                        summary = summary_text[:300]  # Limite la taille
        
        except Exception as e:
            st.warning(f"Erreur lors du parsing de l'évaluation : {e}")
        
        return pertinence_level, summary, score

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
                            except:
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
                continue
    
    st.success(f"✅ {len(all_articles)} articles récupérés au total")
    return all_articles

# --- Interface Streamlit optimisée ---

st.set_page_config(
    page_title="Veille Réglementaire Food Safety", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Veille Réglementaire - Sécurité Alimentaire")
st.markdown("*Application optimisée pour consultants et auditeurs food safety*")
st.markdown("✨ **Nouveau** : Extraction optimisée du contenu RSS (content:encoded pour Health BE)")

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
    
    # Information sur l'extraction de contenu
    with st.expander("ℹ️ Extraction de contenu"):
        st.markdown("""
        **Ordre de priorité pour le contenu:**
        1. `content:encoded` (plus détaillé, notamment Health BE)
        2. `summary/description` (fallback standard)
        3. `title` (fallback minimal)
        
        **Health BE** : Utilise `content:encoded` avec HTML riche
        """)
    
    clean_html = st.session_state.get('clean_html', True)

# --- Profil utilisateur (simplifié) ---
with st.expander("🏢 Profil d'Activité", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        selected_product_types = st.multiselect(
            "Types de Produits",
            PRODUCT_TYPES,
            default=["Autre"]
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
if st.button("🚀 Lancer la Veille", type="primary", use_container_width=True):
    if not groq_api_key:
        st.error("Clé API Groq requise pour l'évaluation de pertinence")
        st.stop()
    
    if start_date > end_date:
        st.error("La date de fin doit être postérieure à la date de début")
        st.stop()
    
    # Récupération des articles
    all_articles = fetch_all_articles(FRENCH_EU_RSS_FEEDS, start_date, end_date, clean_html)
    
    if not all_articles:
        st.warning("Aucun article trouvé dans la période spécifiée")
        st.stop()
    
    # Limitation du nombre d'articles à évaluer
    all_articles = all_articles[:max_articles]
    
    st.info(f"🔍 Évaluation de {len(all_articles)} articles...")
    
    # Évaluation des articles
    evaluator = ArticleEvaluator(groq_api_key)
    evaluated_articles = []
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for idx, article in enumerate(all_articles):
        progress_text.text(f"Évaluation {idx+1}/{len(all_articles)}: {article['source']}")
        
        pertinence_level, evaluation_summary, score = evaluator.evaluate_pertinence(
            article['title'],
            article['summary'],
            user_context
        )
        
        # Filtrage basé sur le score minimum
        if score >= min_pertinence_score:
            evaluated_articles.append({
                "Source": article['source'],
                "Titre": article['title'],
                "Résumé": article['summary'],
                "Lien": article['link'],
                "Date de Publication": article['published'].strftime('%Y-%m-%d'),
                "Niveau de Pertinence": pertinence_level,
                "Score": score,
                "Évaluation de la Pertinence": evaluation_summary,
                "raw_content": article.get('raw_content', '')  # Garde le contenu brut pour debug
            })
        
        progress_bar.progress((idx + 1) / len(all_articles))
    
    progress_text.empty()
    progress_bar.empty()
    
    # Affichage des résultats
    if evaluated_articles:
        # Tri par score décroissant
        evaluated_articles.sort(key=lambda x: x['Score'], reverse=True)
        
        st.success(f"✅ {len(evaluated_articles)} articles pertinents trouvés (score ≥ {min_pertinence_score})")
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        
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
        
        # AFFICHAGE SIMPLE - AUCUN data_editor !
        st.subheader("📊 Articles Sélectionnés")
        
        # 1. Vue d'ensemble avec tableau simple
        st.markdown("### 📋 Vue d'ensemble")
        
        # Préparation simple des données
        simple_data = []
        for article in evaluated_articles:
            score_visual = f"⭐ {article['Score']}/100 " + ("🟢" if article['Score'] >= 80 else "🟡" if article['Score'] >= 60 else "🟠")
            lien_status = "🔗 Disponible" if article['Lien'] != '#' else "❌ Indisponible"
            
            simple_data.append({
                'Source': article['Source'],
                'Titre': article['Titre'][:80] + ("..." if len(article['Titre']) > 80 else ""),
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
            label = f"⭐{article['Score']} - {article['Source']} - {article['Titre'][:50]}..."
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
                            st.markdown(f"**📰 Titre :** {article['Titre']}")
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
        
        # 4. Debug
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
        
        # Téléchargement
        if len(evaluated_articles) > 0:
            st.subheader("💾 Téléchargement")
            
            col1, col2, col3 = st.columns(3)
            
            # Préparation données export
            if selected_indices:
                selected_articles_data = [evaluated_articles[i] for i in selected_indices if i < len(evaluated_articles)]
                export_df = pd.DataFrame(selected_articles_data)
                st.success(f"📌 {len(selected_articles_data)} articles sélectionnés pour l'export")
            else:
                export_df = pd.DataFrame(evaluated_articles)
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
        st.warning(f"❌ Aucun article avec un score ≥ {min_pertinence_score} trouvé. Essayez de réduire le seuil de pertinence.")

# Footer
st.markdown("---")
st.markdown("*Développé pour les professionnels de la sécurité alimentaire européenne*")
