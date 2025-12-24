import streamlit as st
import os

# --- 1. CONFIGURATION SYSTÈME ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# --- 2. IMPORTS ---
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# --- LA CORRECTION CRUCIALE ---
# C'est LE bon chemin d'importation pour la version actuelle
from crewai.tools import tool

# --- 3. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Agent PEA Intelligent", page_icon="📈")
st.title("📈 Assistant PEA Intelligent")
st.markdown("Analyse financière & Sentiment social (X/Reddit)")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Ta clé API Groq", type="password")
    if not api_key:
        st.warning("Entre ta clé pour démarrer.")

# --- 5. DÉFINITION DES OUTILS (Format CrewAI Natif) ---

# Outil 1 : Recherche Web
# On utilise le décorateur de CrewAI (@tool) pour emballer le moteur de LangChain
@tool("Outil Recherche Web")
def recherche_web_tool(query: str):
    """
    Recherche sur internet. Utile pour trouver l'actualité, 
    les avis sur Reddit ou X (Twitter) à propos d'une action.
    """
    # On instancie le moteur de recherche à l'intérieur de la fonction
    search = DuckDuckGoSearchRun()
    return search.run(query)

# Outil 2 : Analyse Bourse
@tool("Outil Analyse Boursiere")
def analyse_bourse_tool(ticker: str):
    """
    Récupère les données boursières via Yahoo Finance (Prix, PER, Dividende).
    L'input doit être un ticker (ex: 'TTE.PA').
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        data = {
            "Entreprise": info.get('longName', ticker),
            "Prix Actuel": info.get('currentPrice', 'N/A'),
            "PER (Price/Earnings)": info.get('forwardPE', 'N/A'),
            "Dividende (%)": (info.get('dividendYield', 0) or 0) * 100,
            "Recommandation Analystes": info.get('recommendationKey', 'Inconnue')
        }
        return str(data)
    except Exception as e:
        return f"Erreur récupération Yahoo : {e}"

# --- 6. MOTEUR DE L'AGENT ---
def run_crew(ticker_symbol):
    # Cerveau
    llm = ChatGroq(
        api_key=api_key,
        model="llama3-70b-8192",
        temperature=0.5
    )

    # Agent 1 : Financier
    analyste = Agent(
        role='Analyste Financier',
        goal='Analyser les fondamentaux',
        backstory="Expert comptable rigoureux.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[analyse_bourse_tool] # C'est maintenant un outil CrewAI valide
    )

    # Agent 2 : Social
    trader = Agent(
        role='Trader Sentiment',
        goal='Sonder l\'opinion sur les réseaux',
        backstory="Expert réseaux sociaux (X, Reddit).",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[recherche_web_tool] # C'est maintenant un outil CrewAI valide
    )

    # Tâches
    task_finance = Task(
        description=f"Donne moi le Prix, le PER et le Dividende de {ticker_symbol}.",
        expected_output="Chiffres clés.",
        agent=analyste
    )

    task_sentiment = Task(
        description=f"Cherche sur le web : 'site:twitter.com {ticker_symbol} avis' et 'site:reddit.com {ticker_symbol}'. Résume l'ambiance.",
        expected_output="Analyse sentiment.",
        agent=trader
    )

    task_synthese = Task(
        description=f"Synthèse finale pour {ticker_symbol} (Achat/Vente ?).",
        expected_output="Recommandation finale.",
        agent=analyste,
        context=[task_finance, task_sentiment]
    )

    crew = Crew(
        agents=[analyste, trader],
        tasks=[task_finance, task_sentiment, task_synthese],
        process=Process.sequential
    )

    return crew.kickoff()

# --- 7. EXÉCUTION ---
ticker_input = st.text_input("Symbole de l'action (ex: TTE.PA)", "TTE.PA")

if st.button("Lancer l'Analyse 🚀"):
    if not api_key:
        st.error("⚠️ Clé API manquante !")
    else:
        with st.status("Analyse en cours...", expanded=True) as status:
            try:
                st.write("🧠 Les agents réfléchissent...")
                resultat = run_crew(ticker_input)
                status.update(label="Terminé !", state="complete", expanded=False)
                st.markdown("### 📊 Résultat")
                st.markdown(resultat)
            except Exception as e:
                st.error(f"Erreur : {e}")
                status.update(label="Erreur", state="error")