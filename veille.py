import streamlit as st
import feedparser
import requests
from datetime import datetime, timedelta
import os
from groq import Groq
import pandas as pd # Import pandas for table and download functionality

# --- Configuration ---
# Catégories prédéfinies pour la saisie de l'utilisateur
PRODUCT_TYPES = ["Produits laitiers", "Viande", "Produits frais", "Produits de boulangerie", "Boissons", "Aliments transformés", "Autre"]
RISK_TYPES = ["Microbiologique", "Chimique", "Physique", "Allergène", "Fraude", "Autre"]
MARKETS = ["UE", "US", "Canada", "Royaume-Uni", "France", "International", "Autre"]

# Flux RSS (sources françaises et européennes)
FRENCH_EU_RSS_FEEDS = {
    "CODEX Hygiene meeting": "https://www.fao.org/fao-who-codexalimentarius/meetings/detail/rss/fr/?meeting=CCFH&session=54",
    "RASFF EU Feed": "https://webgate.ec.europa.eu/rasff-window/backend/public/consumer/rss/all/",
    "EFSA": "https://www.efsa.europa.eu/en/all/rss",
    "EU Food Safety": "https://food.ec.europa.eu/node/2/rss_en",
    "Legifrance Alimentaire": "https://agriculture.gouv.fr/rss.xml",
    "DGCCRF, French Fraud": "https://www.economie.gouv.fr/dgccrf/rss",
    "INRS secu": "https://www.inrs.fr/rss/?feed=actualites",
    "ANSES": "https://www.anses.fr/fr/flux-actualites.rss",
    "Health BE": "https://www.health.belgium.be/fr/rss/news.xml",
}

# Mots-clés prédéfinis sur la sécurité alimentaire pour le contexte de l'API Groq
FOOD_SAFETY_KEYWORDS = [
    "rappel", "contamination", "allergène", "pathogène", "hygiène", "réglementation",
    "norme", "conformité", "audit", "inspection", "danger", "évaluation des risques",
    "maladie d'origine alimentaire", "traçabilité", "HACCP", "GFSI", "pesticide", "additif",
    "emballage", "étiquetage", "microbiologie", "toxicologie", "nouvel aliment", "fraude alimentaire",
    "défense alimentaire", "chaîne d'approvisionnement", "durabilité", "santé publique", "gestion de la sécurité alimentaire"
]

# --- Fonctions ---

@st.cache_data(ttl=3600) # Mettre en cache les données du flux RSS pendant 1 heure
def fetch_rss_feed(url):
    """Récupère et analyse un flux RSS."""
    try:
        feed = feedparser.parse(url)
        return feed.entries
    except Exception as e:
        st.error(f"Erreur lors de la récupération du flux RSS depuis {url} : {e}")
        return []

def get_groq_response(prompt, api_key):
    """Obtient une réponse de l'API Groq."""
    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192", # Utilisation d'un modèle Groq approprié
            temperature=0.2,
            max_tokens=500,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Erreur de communication avec l'API Groq : {e}")
        return None

def evaluate_pertinence(article_title, article_summary, user_context, groq_api_key):
    """Évalue la pertinence d'un article à l'aide de l'API Groq."""
    if not groq_api_key:
        st.warning("Clé API Groq introuvable. L'évaluation de la pertinence sera ignorée.")
        return True, "N/A (Clé API manquante)"

    combined_context = f"Activité de l'utilisateur : {user_context}\nMots-clés de sécurité alimentaire : {', '.join(FOOD_SAFETY_KEYWORDS)}"
    prompt = f"""
    Évaluez la pertinence de l'article suivant pour la sécurité alimentaire, en tenant compte **spécifiquement** de l'activité déclarée par l'utilisateur (types de produits, types de risques, marchés, préoccupations principales) et des mots-clés généraux de sécurité alimentaire.
    Fournissez uniquement un bref résumé de la pertinence et de l'impact potentiel sur les opérations de l'utilisateur, sans poser de question ni inclure de préambule comme "Évaluation de la pertinence de l'article :".

    Titre de l'article : {article_title}
    Résumé de l'article : {article_summary}

    Contexte d'évaluation : {combined_context}

    Résumé de la pertinence et de l'impact potentiel :
    """
    response = get_groq_response(prompt, groq_api_key)
    if response:
        # The Groq API is instructed to return only the summary, so we take it directly.
        return True, response.strip()
    return False, "Impossible d'évaluer la pertinence."

# --- Streamlit UI ---

st.set_page_config(page_title="Veille Réglementaire et Études sur la Sécurité Alimentaire", layout="wide")

st.title("Veille Réglementaire et Études sur la Sécurité Alimentaire")

st.markdown("""
Cette application vous aide à rester informé des réglementations et études en matière de sécurité alimentaire.
Déclarez votre activité, et le système récupérera et évaluera les nouvelles pertinentes provenant de diverses sources.
""")

# --- Déclaration de l'activité de l'utilisateur ---
st.header("1. Déclarez Votre Activité")

with st.expander("Votre Profil d'Entreprise"):
    st.subheader("Types de Produits")
    selected_product_types = st.multiselect(
        "Sélectionnez les types de produits pertinents (prédéfinis)",
        PRODUCT_TYPES,
        default=["Autre"]
    )
    custom_product_types = st.text_area(
        "Ajoutez des types de produits spécifiques non listés (séparés par des virgules)",
        placeholder="ex: Aliments biologiques pour bébés, Snacks sans gluten"
    )

    st.subheader("Types de Risques")
    selected_risk_types = st.multiselect(
        "Sélectionnez les types de risques pertinents (prédéfinis)",
        RISK_TYPES,
        default=["Autre"]
    )
    custom_risk_types = st.text_area(
        "Ajoutez des types de risques spécifiques non listés (séparés par des virgules)",
        placeholder="ex: Norovirus, Aflatoxines, Contamination par le verre"
    )

    st.subheader("Marchés")
    selected_markets = st.multiselect(
        "Sélectionnez vos marchés cibles (prédéfinis)",
        MARKETS,
        default=["International"]
    )
    custom_markets = st.text_area(
        "Ajoutez des marchés spécifiques non listés (séparés par des virgules)",
        placeholder="ex: Japon, Brésil, Moyen-Orient"
    )

    st.subheader("Principales Préoccupations/Mots-clés")
    main_concerns = st.text_area(
        "Décrivez vos principales préoccupations en matière de sécurité alimentaire ou des mots-clés spécifiques d'intérêt (séparés par des virgules)",
        placeholder="ex: PFAS dans les emballages, réglementations sur les nouvelles protéines, contrôle de la Listeria"
    )

# Combiner le contexte utilisateur
user_declared_activity = {
    "product_types": list(set(selected_product_types + [p.strip() for p in custom_product_types.split(',') if p.strip()])),
    "risk_types": list(set(selected_risk_types + [r.strip() for r in custom_risk_types.split(',') if r.strip()])),
    "markets": list(set(selected_markets + [m.strip() for m in custom_markets.split(',') if m.strip()])),
    "main_concerns": [c.strip() for c in main_concerns.split(',') if c.strip()]
}
user_context_string = (
    f"Types de Produits : {', '.join(user_declared_activity['product_types'])}\n"
    f"Types de Risques : {', '.join(user_declared_activity['risk_types'])}\n"
    f"Marchés : {', '.join(user_declared_activity['markets'])}\n"
    f"Principales Préoccupations : {', '.join(user_declared_activity['main_concerns'])}"
)

# --- Période de Veille ---
st.header("2. Définissez la Période de Veille")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Date de Début", datetime.now() - timedelta(weeks=1))
with col2:
    end_date = st.date_input("Date de Fin", datetime.now())

if start_date > end_date:
    st.error("Erreur : La date de fin doit être postérieure à la date de début.")

# --- Saisie de la clé API Groq ---
st.header("3. Configuration de l'API Groq")
st.info("Votre clé API Groq doit être stockée en tant que secret Streamlit. Créez un fichier `.streamlit/secrets.toml` avec `GROQ_API_KEY = \"votre_clé_api_ici\"`.")
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.warning("Clé API Groq introuvable dans les variables d'environnement ou les secrets Streamlit. L'évaluation de la pertinence sera ignorée.")
    st.text_input("Entrez la clé API Groq (pour les tests locaux, ne sera pas sauvegardée)", type="password", key="local_groq_key")
    if st.session_state.get("local_groq_key"):
        groq_api_key = st.session_state["local_groq_key"]

# --- Effectuer la Veille ---
st.header("4. Effectuer la Veille Réglementaire")
if st.button("Démarrer la Veille"):
    if start_date and end_date and start_date <= end_date:
        st.info("Récupération et évaluation des articles. Cela peut prendre un moment...")
        all_articles = []
        for source_name, url in FRENCH_EU_RSS_FEEDS.items(): # Utiliser les flux filtrés
            entries = fetch_rss_feed(url)
            for entry in entries:
                published_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_date = datetime(*entry.published_parsed[:6]).date()
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    published_date = datetime(*entry.updated_parsed[:6]).date()

                if published_date and start_date <= published_date <= end_date:
                    all_articles.append({
                        "source": source_name,
                        "title": entry.title,
                        "summary": entry.summary if hasattr(entry, 'summary') else entry.title,
                        "link": entry.link,
                        "published": published_date
                    })

        st.subheader("Résultats de l'Évaluation")
        if all_articles:
            pertinent_articles_data = []
            for i, article in enumerate(all_articles):
                with st.spinner(f"Évaluation de l'article {i+1}/{len(all_articles)} de {article['source']}..."):
                    is_pertinent, evaluation_summary = evaluate_pertinence(
                        article['title'],
                        article['summary'],
                        user_context_string,
                        groq_api_key
                    )
                    if is_pertinent:
                        pertinent_articles_data.append({
                            "Source": article['source'],
                            "Titre": article['title'],
                            "Résumé": article['summary'],
                            "Lien": article['link'],
                            "Date de Publication": article['published'].strftime('%Y-%m-%d'),
                            "Évaluation de la Pertinence": evaluation_summary
                        })
            
            if pertinent_articles_data:
                st.success(f"Trouvé {len(pertinent_articles_data)} articles pertinents.")
                df = pd.DataFrame(pertinent_articles_data)
                st.dataframe(df, use_container_width=True)

                # Download buttons
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger les résultats en CSV",
                    data=csv_data,
                    file_name="veille_reglementaire.csv",
                    mime="text/csv",
                )

                # For Excel, we need BytesIO
                from io import BytesIO
                excel_buffer = BytesIO()
                df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
                excel_buffer.seek(0)
                st.download_button(
                    label="Télécharger les résultats en Excel",
                    data=excel_buffer,
                    file_name="veille_reglementaire.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            else:
                st.warning("Aucun article pertinent trouvé pour les critères et la période spécifiés.")
        else:
            st.warning("Aucun article trouvé dans la plage de dates spécifiée à partir des flux RSS.")
    else:
        st.error("Veuillez vous assurer que les dates de début et de fin sont valides.")

st.markdown("---")
st.caption("Développé par Roo pour la Veille Réglementaire sur la Sécurité Alimentaire.")
