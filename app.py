import streamlit as st
import os

# --- 1. CONFIGURATION SYSTÈME ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

# --- 2. IMPORTS ---
import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

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

# --- 5. OUTILS ---

@tool("Outil Recherche Web")
def recherche_web_tool(query: str):
    """Recherche sur internet (X, Reddit, News)."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

@tool("Outil Analyse Boursiere")
def analyse_bourse_tool(ticker: str):
    """Récupère les données boursières Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        data = {
            "Entreprise": info.get('longName', ticker),
            "Prix": info.get('currentPrice', 'N/A'),
            "PER": info.get('forwardPE', 'N/A'),
            "Dividende (%)": (info.get('dividendYield', 0) or 0) * 100,
            "Recommandation": info.get('recommendationKey', 'Inconnue')
        }
        return str(data)
    except Exception as e:
        return f"Erreur Yahoo : {e}"

# --- 6. MOTEUR ---
def run_crew(ticker_symbol):
    
    # --- FIX LITELLM ---
    # On injecte la clé directement dans l'environnement pour que LiteLLM la trouve
    os.environ["GROQ_API_KEY"] = api_key
    
    # Définition du LLM via LiteLLM (syntaxe groq/modele)
    my_llm = LLM(
        model="groq/llama3-70b-8192",
        temperature=0.5
    )

    # Agents
    analyste = Agent(
        role='Analyste Financier',
        goal='Analyser les fondamentaux',
        backstory="Expert comptable rigoureux.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[analyse_bourse_tool]
    )

    trader = Agent(
        role='Trader Sentiment',
        goal='Sonder l\'opinion sur les réseaux',
        backstory="Expert réseaux sociaux (X, Reddit).",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[recherche_web_tool]
    )

    # Tâches
    task_finance = Task(
        description=f"Donne les chiffres clés (Prix, PER, Dividende) pour {ticker_symbol}.",
        expected_output="Synthèse financière.",
        agent=analyste
    )

    task_sentiment = Task(
        description=f"Cherche l'avis des gens sur {ticker_symbol} via 'site:twitter.com {ticker_symbol}' et Reddit.",
        expected_output="Synthèse sentiment.",
        agent=trader
    )

    task_synthese = Task(
        description=f"Synthèse finale : Faut-il investir dans {ticker_symbol} pour un PEA ? Argumente.",
        expected_output="Rapport final.",
        agent=analyste,
        context=[task_finance, task_sentiment]
    )

    # Crew
    crew = Crew(
        agents=[analyste, trader],
        tasks=[task_finance, task_sentiment, task_synthese],
        process=Process.sequential,
        memory=False, 
        verbose=True
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