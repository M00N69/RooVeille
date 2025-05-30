import streamlit as st
import feedparser
import requests
from datetime import datetime, timedelta
import os
from groq import Groq
import pandas as pd
from io import BytesIO

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
    """Évalue la pertinence d'un article à l'aide de l'API Groq et classe son niveau."""
    if not groq_api_key:
        st.warning("Clé API Groq introuvable. L'évaluation de la pertinence sera ignorée.")
        return "Non pertinent", "N/A (Clé API manquante)"

    combined_context = f"Activité de l'utilisateur : {user_context}\nMots-clés de sécurité alimentaire : {', '.join(FOOD_SAFETY_KEYWORDS)}"
    prompt = f"""
    Évaluez la pertinence de l'article suivant pour la sécurité alimentaire, en tenant compte **spécifiquement** de l'activité déclarée par l'utilisateur (types de produits, types de risques, marchés, préoccupations principales) et des mots-clés généraux de sécurité alimentaire.

    Classez la pertinence comme "Très pertinent", "Modérément pertinent" ou "Non pertinent".
    Fournissez ensuite un bref résumé de la pertinence et de l'impact potentiel sur les opérations de l'utilisateur.

    Format de la réponse :
    Pertinence: [Très pertinent/Modérément pertinent/Non pertinent]
    Résumé: [Bref résumé de la pertinence et de l'impact potentiel]

    Titre de l'article : {article_title}
    Résumé de l'article : {article_summary}

    Contexte d'évaluation : {combined_context}
    """
    response = get_groq_response(prompt, groq_api_key)
    if response:
        pertinence_level = "Non pertinent"
        summary = "Impossible d'évaluer la pertinence."
        
        lines = response.split('\n')
        for line in lines:
            if line.startswith("Pertinence:"):
                pertinence_level = line.replace("Pertinence:", "").strip()
            elif line.startswith("Résumé:"):
                summary = line.replace("Résumé:", "").strip()
        
        return pertinence_level, summary
    return "Non pertinent", "Impossible d'évaluer la pertinence."

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
            highly_pertinent_articles_data = []
            moderately_pertinent_articles_data = []
            
            for i, article in enumerate(all_articles):
                with st.spinner(f"Évaluation de l'article {i+1}/{len(all_articles)} de {article['source']}..."):
                    pertinence_level, evaluation_summary = evaluate_pertinence(
                        article['title'],
                        article['summary'],
                        user_context_string,
                        groq_api_key
                    )
                    
                    article_data = {
                        "Source": article['source'],
                        "Titre": article['title'],
                        "Résumé": article['summary'],
                        "Lien": article['link'],
                        "Date de Publication": article['published'].strftime('%Y-%m-%d'),
                        "Évaluation de la Pertinence": evaluation_summary
                    }

                    if pertinence_level == "Très pertinent":
                        highly_pertinent_articles_data.append(article_data)
                    elif pertinence_level == "Modérément pertinent":
                        moderately_pertinent_articles_data.append(article_data)
            
            # Display Highly Pertinent Articles
            if highly_pertinent_articles_data:
                st.markdown("### Articles Très Pertinents")
                df_highly_pertinent = pd.DataFrame(highly_pertinent_articles_data)
                st.session_state['highly_pertinent_selection'] = st.data_editor(
                    df_highly_pertinent,
                    key="highly_pertinent_editor",
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Lien": st.column_config.LinkColumn("Lien", display_text="Ouvrir l'article"),
                        "Résumé": st.column_config.Column("Résumé", width="large"),
                        "Évaluation de la Pertinence": st.column_config.Column("Évaluation de la Pertinence", width="large"),
                    },
                    num_rows="dynamic",
                    selection_mode="multi-row"
                )
                
            # Option to display Moderately Pertinent Articles
            display_moderately_pertinent = st.checkbox("Afficher les articles modérément pertinents")
            if display_moderately_pertinent and moderately_pertinent_articles_data:
                st.markdown("### Articles Modérément Pertinents")
                df_moderately_pertinent = pd.DataFrame(moderately_pertinent_articles_data)
                st.session_state['moderately_pertinent_selection'] = st.data_editor(
                    df_moderately_pertinent,
                    key="moderately_pertinent_editor",
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Lien": st.column_config.LinkColumn("Lien", display_text="Ouvrir l'article"),
                        "Résumé": st.column_config.Column("Résumé", width="large"),
                        "Évaluation de la Pertinence": st.column_config.Column("Évaluation de la Pertinence", width="large"),
                    },
                    num_rows="dynamic",
                    selection_mode="multi-row"
                )
            elif display_moderately_pertinent and not moderately_pertinent_articles_data:
                st.info("Aucun article modérément pertinent trouvé pour les critères et la période spécifiés.")


            if highly_pertinent_articles_data or (display_moderately_pertinent and moderately_pertinent_articles_data):
                st.markdown("---")
                st.subheader("Télécharger les articles sélectionnés")
                
                selected_rows_data = []
                if 'highly_pertinent_selection' in st.session_state and st.session_state['highly_pertinent_selection']['selection']['rows']:
                    selected_indices = st.session_state['highly_pertinent_selection']['selection']['rows']
                    selected_rows_data.extend(df_highly_pertinent.iloc[selected_indices].to_dict(orient='records'))
                
                if display_moderately_pertinent and 'moderately_pertinent_selection' in st.session_state and st.session_state['moderately_pertinent_selection']['selection']['rows']:
                    selected_indices = st.session_state['moderately_pertinent_selection']['selection']['rows']
                    selected_rows_data.extend(df_moderately_pertinent.iloc[selected_indices].to_dict(orient='records'))

                if selected_rows_data:
                    df_selected = pd.DataFrame(selected_rows_data)
                    
                    csv_data = df_selected.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Télécharger la sélection en CSV",
                        data=csv_data,
                        file_name="veille_reglementaire_selection.csv",
                        mime="text/csv",
                    )

                    excel_buffer = BytesIO()
                    df_selected.to_excel(excel_buffer, index=False, engine='xlsxwriter')
                    excel_buffer.seek(0)
                    st.download_button(
                        label="Télécharger la sélection en Excel",
                        data=excel_buffer,
                        file_name="veille_reglementaire_selection.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                else:
                    st.info("Sélectionnez des articles dans les tableaux ci-dessus pour les télécharger.")

            elif not highly_pertinent_articles_data and not (display_moderately_pertinent and moderately_pertinent_articles_data):
                st.warning("Aucun article pertinent trouvé pour les critères et la période spécifiés.")
        else:
            st.warning("Aucun article trouvé dans la plage de dates spécifiée à partir des flux RSS.")
    else:
        st.error("Veuillez vous assurer que les dates de début et de fin sont valides.")

st.markdown("---")
st.caption("Développé par Roo pour la Veille Réglementaire sur la Sécurité Alimentaire.")
