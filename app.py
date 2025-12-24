import streamlit as st
import os
import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# --- 1. CONFIGURATION ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA" # Bloque OpenAI

# --- 2. INTERFACE ---
st.set_page_config(page_title="Agent PEA Gemini", page_icon="💎", layout="wide")
st.title("💎 Assistant PEA (Gemini Pro)")
st.markdown("Analyse financière & Sentiment social")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Ta clé Google API (AIza...)", type="password")
    if not api_key:
        st.warning("Entre ta clé.")

# --- 4. OUTILS ---

@tool("Recherche Web")
def recherche_web_tool(query: str):
    """Recherche sur internet (News, Sentiment)."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results: return "Pas de résultat."
            return "\n".join([f"- {r['body']}" for r in results])
    except Exception as e:
        return "Erreur recherche."

@tool("Bourse Yahoo")
def analyse_bourse_tool(ticker: str):
    """Données financières."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        data = {
            "Nom": info.get('longName', ticker),
            "Prix": info.get('currentPrice', 'N/A'),
            "PER": info.get('forwardPE', 'N/A'),
            "Div": (info.get('dividendYield', 0) or 0) * 100,
            "Avis": info.get('recommendationKey', 'Inconnue')
        }
        return str(data)
    except Exception:
        return "Erreur données."

# --- 5. MOTEUR ---
def run_crew(ticker_symbol):
    
    # Injection des clés
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # --- CHANGEMENT ICI : Modèle plus standard ---
    my_llm = LLM(
        model="gemini/gemini-pro", # On revient au standard absolu
        api_key=api_key,
        temperature=0.3
    )

    # Agents
    analyste = Agent(
        role='Analyste',
        goal='Chiffres clés',
        backstory="Expert comptable.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[analyse_bourse_tool]
    )

    trader = Agent(
        role='Trader',
        goal='Sentiment web',
        backstory="Expert réseaux.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[recherche_web_tool]
    )

    # Tâches
    task_finance = Task(
        description=f"Donne les chiffres (Prix, PER, Dividende) de {ticker_symbol}.",
        expected_output="Synthèse financière.",
        agent=analyste
    )

    task_sentiment = Task(
        description=f"Cherche l'avis sur {ticker_symbol} (Web/Reddit).",
        expected_output="Synthèse sentiment.",
        agent=trader
    )

    task_synthese = Task(
        description=f"Conclusion PEA pour {ticker_symbol}. Achat/Vente ? Court.",
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

# --- 6. EXÉCUTION ---
ticker_input = st.text_input("Action (ex: TTE.PA)", "TTE.PA")

if st.button("Lancer 🚀"):
    if not api_key:
        st.error("Clé manquante !")
    else:
        with st.status("Gemini travaille...", expanded=True) as status:
            try:
                st.write("💎 Initialisation...")
                resultat = run_crew(ticker_input)
                status.update(label="✅ Terminé !", state="complete", expanded=False)
                st.markdown("### 📊 Résultat")
                st.markdown(resultat)
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                status.update(label="Erreur", state="error")