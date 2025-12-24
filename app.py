import streamlit as st
import os

# --- 1. CONFIGURATION SYSTÈME ---
# Désactive la télémétrie pour éviter les erreurs de threads
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
# Force une fausse clé OpenAI pour empêcher CrewAI de la chercher
os.environ["OPENAI_API_KEY"] = "NA"

# --- 2. IMPORTS ---
import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# --- 3. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Agent PEA Intelligent", page_icon="📈", layout="wide")

st.title("📈 Assistant PEA Intelligent (Llama 3.3)")
st.markdown("""
Cet agent utilise le tout dernier modèle **Llama 3.3 70B** via Groq (Gratuit).
Il analyse les données financières (Yahoo) et le sentiment social (Web).
""")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Ta clé API Groq", type="password")
    if not api_key:
        st.warning("Entre ta clé pour démarrer.")
        st.markdown("[Obtenir une clé Groq ici](https://console.groq.com/keys)")

# --- 5. DÉFINITION DES OUTILS ---

@tool("Outil Recherche Web")
def recherche_web_tool(query: str):
    """
    Recherche sur internet (X, Reddit, News).
    Utile pour connaître le sentiment du marché.
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)

@tool("Outil Analyse Boursiere")
def analyse_bourse_tool(ticker: str):
    """
    Récupère les données boursières Yahoo Finance.
    Input: Ticker (ex: 'TTE.PA').
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        data = {
            "Entreprise": info.get('longName', ticker),
            "Prix Actuel": info.get('currentPrice', 'N/A'),
            "PER (Price/Earnings)": info.get('forwardPE', 'N/A'),
            "Dividende (%)": (info.get('dividendYield', 0) or 0) * 100,
            "Recommandation Analystes": info.get('recommendationKey', 'Inconnue'),
            "Secteur": info.get('sector', 'N/A')
        }
        return str(data)
    except Exception as e:
        return f"Erreur Yahoo : {e}"

# --- 6. MOTEUR DE L'AGENT ---
def run_crew(ticker_symbol):
    
    # Injection de la clé pour LiteLLM
    os.environ["GROQ_API_KEY"] = api_key
    
    # --- LE NOUVEAU CERVEAU ---
    # On utilise le modèle Llama 3.3 (Le plus performant actuellement)
    my_llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.5
    )

    # Agent 1 : Analyste Financier
    analyste = Agent(
        role='Analyste Financier Senior',
        goal='Analyser la santé financière et la rentabilité',
        backstory="Expert comptable rigoureux, obsédé par les dividendes et le PER.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[analyse_bourse_tool]
    )

    # Agent 2 : Trader Sentiment
    trader = Agent(
        role='Analyste Sentiment de Marché',
        goal='Sonder l\'opinion publique sur le Web',
        backstory="Expert des réseaux sociaux, capable de détecter la peur ou l'euphorie.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[recherche_web_tool]
    )

    # Tâches
    task_finance = Task(
        description=f"Analyse les fondamentaux de {ticker_symbol} (Prix, PER, Dividende). Est-ce une action solide ?",
        expected_output="Synthèse des chiffres clés.",
        agent=analyste
    )

    task_sentiment = Task(
        description=f"Recherche 'site:twitter.com {ticker_symbol}' et 'site:reddit.com {ticker_symbol} avis'. Quelle est l'ambiance ?",
        expected_output="Synthèse du sentiment social.",
        agent=trader
    )

    task_synthese = Task(
        description=f"En te basant sur les chiffres et le sentiment, rédige une recommandation finale pour {ticker_symbol} (PEA). Argumente.",
        expected_output="Rapport final complet en markdown.",
        agent=analyste,
        context=[task_finance, task_sentiment]
    )

    # Équipe (Mémoire désactivée pour vitesse & gratuité)
    crew = Crew(
        agents=[analyste, trader],
        tasks=[task_finance, task_sentiment, task_synthese],
        process=Process.sequential,
        memory=False,
        verbose=True
    )

    return crew.kickoff()

# --- 7. EXÉCUTION ---
ticker_input = st.text_input("Symbole de l'action (ex: TTE.PA, MC.PA)", "TTE.PA")

if st.button("Lancer l'Analyse 🚀"):
    if not api_key:
        st.error("⚠️ Clé API manquante ! Regarde dans la colonne de gauche.")
    else:
        with st.status("🚀 Les agents Llama 3.3 travaillent...", expanded=True) as status:
            try:
                st.write("🔍 Récupération des données...")
                resultat = run_crew(ticker_input)
                status.update(label="✅ Terminé !", state="complete", expanded=False)
                
                st.divider()
                st.markdown("### 📊 Rapport Final")
                st.markdown(resultat)
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                status.update(label="❌ Erreur", state="error")