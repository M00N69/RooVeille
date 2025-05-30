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

# --- Configuration ---
PRODUCT_TYPES = ["Produits laitiers", "Viande", "Produits frais", "Produits de boulangerie", "Boissons", "Aliments transformés", "Autre"]
RISK_TYPES = ["Microbiologique", "Chimique", "Physique", "Allergène", "Fraude", "Autre"]
MARKETS = ["UE", "US", "Canada", "Royaume-Uni", "France", "International", "Autre"]

# Flux RSS optimisés
FRENCH_EU_RSS_FEEDS = {
    "RASFF EU Feed": "https://webgate.ec.europa.eu/rasff-window/backend/public/consumer/rss/all/",
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
        pertinence_level = "Non pertinent"
        summary = "Impossible d'évaluer"
        score = 0
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Pertinence:"):
                pertinence_level = line.replace("Pertinence:", "").strip()
            elif line.startswith("Score:"):
                try:
                    score = int(line.replace("Score:", "").strip())
                except ValueError:
                    score = 0
            elif line.startswith("Résumé:"):
                summary = line.replace("Résumé:", "").strip()
        
        return pertinence_level, summary, score

@st.cache_data(ttl=1800)  # Cache réduit à 30 minutes pour plus de fraîcheur
def fetch_rss_feed(url: str) -> List[Dict]:
    """Récupère les entrées RSS avec gestion d'erreur améliorée."""
    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            st.warning(f"Problème de parsing pour {url}")
        return feed.entries
    except Exception as e:
        st.error(f"Erreur RSS {url}: {e}")
        return []

def fetch_all_articles(feeds: Dict[str, str], start_date: datetime.date, end_date: datetime.date) -> List[Dict]:
    """Récupère tous les articles en parallèle pour plus d'efficacité."""
    all_articles = []
    
    with st.spinner("Récupération des flux RSS..."):
        progress_bar = st.progress(0)
        
        for idx, (source_name, url) in enumerate(feeds.items()):
            try:
                entries = fetch_rss_feed(url)
                
                for entry in entries:
                    published_date = None
                    
                    # Gestion améliorée des dates
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6]).date()
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_date = datetime(*entry.updated_parsed[:6]).date()
                    elif hasattr(entry, 'published'):
                        try:
                            published_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z').date()
                        except:
                            published_date = datetime.now().date()
                    
                    if published_date and start_date <= published_date <= end_date:
                        all_articles.append({
                            "source": source_name,
                            "title": entry.title,
                            "summary": getattr(entry, 'summary', entry.title),
                            "link": entry.link,
                            "published": published_date
                        })
                        
                progress_bar.progress((idx + 1) / len(feeds))
                
            except Exception as e:
                st.warning(f"Erreur lors de la récupération de {source_name}: {e}")
    
    return all_articles

def create_enhanced_dataframe(articles_data: List[Dict]) -> pd.DataFrame:
    """Crée un DataFrame optimisé pour l'affichage avec retours à la ligne."""
    if not articles_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(articles_data)
    
    # Formatage amélioré des colonnes
    df['Titre_Display'] = df['Titre'].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)
    df['Résumé_Display'] = df['Résumé'].apply(lambda x: x[:200] + "..." if len(x) > 200 else x)
    df['Score_Display'] = df['Score'].apply(lambda x: f"{x}/100")
    
    # Réorganisation des colonnes pour l'affichage
    display_df = df[['Source', 'Titre_Display', 'Résumé_Display', 'Score_Display', 
                     'Date de Publication', 'Évaluation de la Pertinence', 'Lien']].copy()
    
    display_df.columns = ['Source', 'Titre', 'Résumé', 'Score', 'Date', 'Évaluation', 'Lien']
    
    return display_df

# --- Interface Streamlit optimisée ---

st.set_page_config(
    page_title="Veille Réglementaire Food Safety", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Veille Réglementaire - Sécurité Alimentaire")
st.markdown("*Application optimisée pour consultants et auditeurs food safety*")

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
    all_articles = fetch_all_articles(FRENCH_EU_RSS_FEEDS, start_date, end_date)
    
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
                "Évaluation de la Pertinence": evaluation_summary
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
        
        # Tableau interactif amélioré
        st.subheader("📊 Articles Sélectionnés")
        
        df_display = create_enhanced_dataframe(evaluated_articles)
        
        selected_articles = st.data_editor(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Lien": st.column_config.LinkColumn("🔗 Lien", display_text="Ouvrir"),
                "Titre": st.column_config.TextColumn("📰 Titre", width="large"),
                "Résumé": st.column_config.TextColumn("📝 Résumé", width="large"),
                "Score": st.column_config.TextColumn("⭐ Score", width="small"),
                "Évaluation": st.column_config.TextColumn("🔍 Évaluation", width="large"),
                "Source": st.column_config.TextColumn("📡 Source", width="medium"),
                "Date": st.column_config.DateColumn("📅 Date", width="small")
            },
            key="articles_selection"
        )
        
        # Téléchargement des résultats
        if len(evaluated_articles) > 0:
            st.subheader("💾 Téléchargement")
            
            col1, col2 = st.columns(2)
            
            # Préparation des données pour export
            export_df = pd.DataFrame(evaluated_articles)
            
            with col1:
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv_data,
                    file_name=f"veille_food_safety_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, sheet_name='Veille Food Safety', index=False)
                    
                    # Formatage Excel
                    workbook = writer.book
                    worksheet = writer.sheets['Veille Food Safety']
                    
                    # Format des colonnes
                    worksheet.set_column('A:A', 15)  # Source
                    worksheet.set_column('B:B', 50)  # Titre  
                    worksheet.set_column('C:C', 60)  # Résumé
                    worksheet.set_column('D:D', 40)  # Lien
                    worksheet.set_column('E:E', 12)  # Date
                    worksheet.set_column('F:F', 20)  # Niveau
                    worksheet.set_column('G:G', 8)   # Score
                    worksheet.set_column('H:H', 80)  # Évaluation
                
                excel_buffer.seek(0)
                st.download_button(
                    label="📥 Télécharger Excel",
                    data=excel_buffer,
                    file_name=f"veille_food_safety_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    else:
        st.warning(f"❌ Aucun article avec un score ≥ {min_pertinence_score} trouvé. Essayez de réduire le seuil de pertinence.")

# Footer
st.markdown("---")
st.markdown("*Développé pour les professionnels de la sécurité alimentaire européenne*")
