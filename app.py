import streamlit as st
import os
import time

# --- 1. CONFIGURATION SYSTÈME ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

# --- 2. IMPORTS ---
import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
# On importe directement la classe de base pour éviter les erreurs de wrapper
from duckduckgo_search import DDGS

# --- 3. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Agent PEA Intelligent", page_icon="📈")
st.title("📈 Assistant PEA (Modèle Rapide)")
st.markdown("Analyse financière & Sentiment social - **Optimisé pour Groq Free Tier**")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Ta clé API Groq", type="password")
    if not api_key:
        st.warning("Entre ta clé pour démarrer.")

# --- 5. DÉFINITION DES OUTILS ---

@tool("Outil Recherche Web")
def recherche_web_tool(query: str):
    """
    Recherche sur internet (X, Reddit, News).
    """
    try:
        # On utilise directement la librairie sous-jacente pour être plus robuste
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
            if not results:
                return "Aucun résultat trouvé."
            # On formate proprement les résultats
            formatted_results = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
            return formatted_results
    except Exception as e:
        return f"Erreur de recherche : {e}"

@tool("Outil Analyse Boursiere")
def analyse_bourse_tool(ticker: str):
    """
    Récupère les données boursières Yahoo Finance.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        data = {
            "Entreprise": info.get('longName', ticker),
            "Prix Actuel": info.get('currentPrice', 'N/A'),
            "PER": info.get('forwardPE', 'N/A'),
            "Dividende (%)": (info.get('dividendYield', 0) or 0) * 100,
            "Recommandation": info.get('recommendationKey', 'Inconnue')
        }
        return str(data)
    except Exception as e:
        return f"Erreur Yahoo : {e}"

# --- 6. MOTEUR DE L'AGENT ---
def run_crew(ticker_symbol):
    
    os.environ["GROQ_API_KEY"] = api_key
    
    # --- CHANGEMENT DE MODÈLE (CRUCIAL) ---
    # On utilise le modèle 8B (8 milliards de paramètres).
    # Il est beaucoup plus léger et évite l'erreur "Rate Limit".
    my_llm = LLM(
        model="groq/llama-3.1-8b-instant",
        temperature=0.3
    )

    # Agents
    analyste = Agent(
        role='Analyste Financier',
        goal='Synthétiser les données financières',
        backstory="Expert comptable factuel.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[analyse_bourse_tool]
    )

    trader = Agent(
        role='Analyste Sentiment',
        goal='Trouver le sentiment sur le web',
        backstory="Expert réseaux sociaux.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[recherche_web_tool]
    )

    # Tâches
    task_finance = Task(
        description=f"Donne les chiffres clés (Prix, PER, Dividende) pour {ticker_symbol}.",
        expected_output="Synthèse des chiffres.",
        agent=analyste
    )

    task_sentiment = Task(
        description=f"Cherche sur le web ce qu'on dit de {ticker_symbol} (Twitter/Reddit).",
        expected_output="Résumé du sentiment (Positif/Négatif).",
        agent=trader
    )

    task_synthese = Task(
        description=f"Conclusion pour {ticker_symbol} (PEA). Achat ou pas ? Sois bref.",
        expected_output="Conseil final court.",
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
        with st.status("Analyse en cours (Mode Rapide)...", expanded=True) as status:
            try:
                st.write("🧠 Agents activés...")
                resultat = run_crew(ticker_input)
                status.update(label="Terminé !", state="complete", expanded=False)
                st.markdown("### 📊 Résultat")
                st.markdown(resultat)
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                status.update(label="Erreur", state="error")