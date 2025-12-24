import streamlit as st
import os

# --- DISABLE TELEMETRY ---
# Cela évite les erreurs de "Threads" rouges dans la console
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

# --- IMPORT OFFICIEL ---
# Maintenant que crewai-tools est dans requirements.txt, ça va marcher !
from crewai_tools import DuckDuckGoSearchTool, tool

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Agent PEA Intelligent", page_icon="📈")

st.title("📈 Assistant PEA Intelligent")
st.markdown("Analyse financière & Sentiment social (X/Reddit)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Ta clé API Groq", type="password")
    st.info("Récupère ta clé sur console.groq.com")

# --- FONCTIONS & OUTILS ---
def run_analysis(ticker):
    # 1. Configurer l'API
    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.5)

    # 2. Outils
    search_tool = DuckDuckGoSearchTool()

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

    # 3. Agents
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
        backstory="Expert des réseaux sociaux et de la psychologie de marché.",
        llm=llm,
        tools=[search_tool],
        verbose=True,
        allow_delegation=False
    )

    # 4. Tâches
    task_finance = Task(
        description=f"Analyse les chiffres clés de {ticker} (Prix, PER, Dividende).",
        expected_output="Rapport financier court.",
        agent=analyste
    )

    task_social = Task(
        description=f"Cherche le sentiment sur {ticker} via 'site:twitter.com {ticker}' et Reddit.",
        expected_output="Synthèse de l'humeur sociale.",
        agent=trader_social
    )

    task_final = Task(
        description=f"Synthétise les chiffres et l'humeur pour {ticker}. Donne une recommandation (Achat/Vente/Attente).",
        expected_output="Rapport final structuré en Markdown.",
        agent=analyste,
        context=[task_finance, task_social]
    )

    # 5. Crew
    crew = Crew(
        agents=[analyste, trader_social],
        tasks=[task_finance, task_social, task_final],
        process=Process.sequential
    )

    return crew.kickoff()

# --- INTERFACE PRINCIPALE ---
ticker_input = st.text_input("Symbole de l'action (ex: TTE.PA, AI.PA, MC.PA)", "TTE.PA")

if st.button("Lancer l'analyse 🚀"):
    if not api_key:
        st.error("Merci d'entrer une clé API Groq dans la barre latérale.")
    else:
        with st.status("L'agent travaille... (Regarde les détails ici)", expanded=True) as status:
            st.write("🤖 Initialisation des agents...")
            try:
                resultat = run_analysis(ticker_input)
                st.write("✅ Analyse terminée !")
                status.update(label="Mission accomplie !", state="complete", expanded=False)
                
                st.divider()
                st.subheader("Rapport Final")
                st.markdown(resultat)
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                st.info("Astuce : Si l'erreur persiste, essaie de redémarrer l'app (Reboot App).")
                status.update(label="Erreur", state="error")