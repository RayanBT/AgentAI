import streamlit as st
import os

# --- DISABLE TELEMETRY ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

import yfinance as yf

# 1. IMPORTS CREWAI
from crewai import Agent, Task, Crew, Process
# On n'importe QUE 'tool' d'ici. On laisse tomber DuckDuckGoSearchTool qui est buggé.
from crewai_tools import tool 

# 2. IMPORTS LANGCHAIN & GROQ
from langchain_groq import ChatGroq
# On utilise le moteur de recherche de LangChain (plus stable)
from langchain_community.tools import DuckDuckGoSearchRun

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Agent PEA Intelligent", page_icon="📈")
st.title("📈 Assistant PEA Intelligent")
st.markdown("Analyse financière & Sentiment social (X/Reddit)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Ta clé API Groq", type="password")
    if not api_key:
        st.warning("⚠️ Clé API manquante")

# --- FONCTIONS ---
def run_analysis(ticker):
    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.5)

    # --- CRÉATION DES OUTILS "FAITS MAISON" ---

    # 1. Outil de Recherche (Remplacement manuel de l'outil buggé)
    @tool("Outil Recherche Web")
    def custom_search_tool(query: str):
        """Utilise ce moteur pour chercher des actualités, des avis sur X/Reddit."""
        search_engine = DuckDuckGoSearchRun()
        return search_engine.run(query)

    # 2. Outil Bourse
    @tool("Outil Analyse Boursiere")
    def stock_analysis_tool(ticker_symbol: str):
        """Récupère les données financières (Prix, PER, Dividende)."""
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            data = {
                "Nom": info.get('longName'),
                "Prix": info.get('currentPrice'),
                "PER": info.get('forwardPE'),
                "Dividende (%)": (info.get('dividendYield', 0) or 0) * 100,
                "Recommandation": info.get('recommendationKey')
            }
            return str(data)
        except Exception as e:
            return f"Erreur: {str(e)}"

    # --- AGENTS ---
    analyste = Agent(
        role='Analyste Financier',
        goal='Analyser les fondamentaux',
        backstory="Expert comptable rigoureux.",
        llm=llm,
        tools=[stock_analysis_tool],
        verbose=True,
        allow_delegation=False
    )

    trader_social = Agent(
        role='Expert Sentiment Social',
        goal='Analyser X et Reddit',
        backstory="Expert réseaux sociaux.",
        llm=llm,
        tools=[custom_search_tool], # On utilise notre outil manuel
        verbose=True,
        allow_delegation=False
    )

    # --- TÂCHES ---
    task_finance = Task(
        description=f"Analyse les chiffres de {ticker} (Prix, PER, Dividende).",
        expected_output="Rapport financier court.",
        agent=analyste
    )

    task_social = Task(
        description=f"Cherche le sentiment sur {ticker} via 'site:twitter.com {ticker}' et Reddit.",
        expected_output="Synthèse humeur.",
        agent=trader_social
    )

    task_final = Task(
        description=f"Synthèse finale pour {ticker}. Recommandation (Achat/Vente ?).",
        expected_output="Rapport final structuré.",
        agent=analyste,
        context=[task_finance, task_social]
    )

    crew = Crew(
        agents=[analyste, trader_social],
        tasks=[task_finance, task_social, task_final],
        process=Process.sequential
    )

    return crew.kickoff()

# --- INTERFACE ---
ticker_input = st.text_input("Symbole de l'action (ex: TTE.PA)", "TTE.PA")

if st.button("Lancer l'analyse 🚀"):
    if not api_key:
        st.error("Entre ta clé API Groq à gauche !")
    else:
        with st.status("Travail en cours...", expanded=True) as status:
            try:
                st.write("🤖 Initialisation...")
                res = run_analysis(ticker_input)
                status.update(label="Terminé !", state="complete", expanded=False)
                st.markdown("### 📊 Rapport Final")
                st.markdown(res)
            except Exception as e:
                st.error(f"Erreur : {e}")
                status.update(label="Erreur", state="error")