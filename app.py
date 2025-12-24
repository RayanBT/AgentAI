import streamlit as st
import os
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from duckduckgo_search import DDGS

# --- 1. CONFIGURATION SYSTÈME ---
# On désactive la télémétrie pour éviter les lignes rouges dans les logs
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# --- 2. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Agent PEA Gemini", page_icon="💎", layout="wide")

st.title("💎 Assistant PEA (Google Gemini)")
st.markdown("""
Cet agent utilise **Gemini 1.5 Flash**.  
Il est gratuit, rapide et possède une grande capacité d'analyse.
""")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Ta clé Google API", type="password")
    if not api_key:
        st.warning("Entre ta clé pour démarrer.")
        st.markdown("[Obtenir une clé Google ici](https://aistudio.google.com/app/apikey)")

# --- 4. DÉFINITION DES OUTILS ---

@tool("Recherche Web")
def recherche_web_tool(query: str):
    """
    Recherche sur internet (X, Reddit, News).
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "Aucun résultat trouvé."
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Erreur de recherche : {e}"

@tool("Bourse Yahoo")
def analyse_bourse_tool(ticker: str):
    """
    Récupère les données financières (Prix, PER, Dividende).
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        data = {
            "Nom": info.get('longName', ticker),
            "Prix": info.get('currentPrice', 'N/A'),
            "Devise": info.get('currency', 'EUR'),
            "PER": info.get('forwardPE', 'N/A'),
            "Dividende (%)": (info.get('dividendYield', 0) or 0) * 100,
            "Recommandation": info.get('recommendationKey', 'Inconnue'),
            "Secteur": info.get('sector', 'N/A')
        }
        return str(data)
    except Exception as e:
        return f"Erreur Yahoo : {e}"

# --- 5. MOTEUR DE L'AGENT ---
def run_crew(ticker_symbol):
    
    # --- CORRECTION CRITIQUE ICI ---
    # On force la clé dans l'environnement système pour que Google la trouve
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Configuration du modèle Gemini
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        verbose=True,
        temperature=0.4,
        google_api_key=api_key
    )

    # Agent 1 : Financier
    analyste = Agent(
        role='Analyste Financier',
        goal='Analyser les chiffres clés',
        backstory="Expert comptable rigoureux.",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[analyse_bourse_tool]
    )

    # Agent 2 : Social
    trader = Agent(
        role='Trader Sentiment',
        goal='Sonder le web',
        backstory="Expert réseaux sociaux.",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[recherche_web_tool]
    )

    # Tâches
    task_finance = Task(
        description=f"Donne les fondamentaux de {ticker_symbol} (Prix, PER, Dividende).",
        expected_output="Synthèse financière.",
        agent=analyste
    )

    task_sentiment = Task(
        description=f"Cherche les avis récents sur {ticker_symbol} (Web/Reddit).",
        expected_output="Synthèse sentiment.",
        agent=trader
    )

    task_synthese = Task(
        description=f"Conclusion pour {ticker_symbol} (PEA). Achat/Vente ? Argumente.",
        expected_output="Rapport final.",
        agent=analyste,
        context=[task_finance, task_sentiment]
    )

    # Lancement
    crew = Crew(
        agents=[analyste, trader],
        tasks=[task_finance, task_sentiment, task_synthese],
        process=Process.sequential,
        memory=False, # Désactivé pour la vitesse
        verbose=True
    )

    return crew.kickoff()

# --- 6. EXÉCUTION ---
ticker_input = st.text_input("Symbole de l'action (ex: TTE.PA)", "TTE.PA")

if st.button("Lancer l'Analyse avec Gemini 🚀"):
    if not api_key:
        st.error("⚠️ Clé manquante !")
    else:
        with st.status("Gemini travaille...", expanded=True) as status:
            try:
                st.write("🧠 Réflexion en cours...")
                resultat = run_crew(ticker_input)
                status.update(label="✅ Terminé !", state="complete", expanded=False)
                
                st.divider()
                st.markdown("### 📊 Rapport Final")
                st.markdown(resultat)
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                status.update(label="Erreur", state="error")