import streamlit as st
import os
import time
from tenacity import retry, wait_fixed, stop_after_attempt # Pour gérer les retries proprement

# --- 1. CONFIGURATION SYSTÈME ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

# --- 2. IMPORTS ---
import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# --- 3. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Agent PEA", page_icon="📈")
st.title("📈 Assistant PEA (Mode Éco)")
st.markdown("Analyse financière & Sentiment - **Optimisé Groq Free**")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Ta clé API Groq", type="password")
    if not api_key:
        st.warning("Entre ta clé.")

# --- 5. DÉFINITION DES OUTILS ---

@tool("Recherche Web")
def recherche_web_tool(query: str):
    """Cherche sur le web (News/Sentiment)."""
    try:
        with DDGS() as ddgs:
            # On réduit à 3 résultats pour économiser des tokens
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "Pas de résultat."
            return "\n".join([f"- {r['body']}" for r in results])
    except Exception as e:
        return "Erreur recherche."

@tool("Bourse Yahoo")
def analyse_bourse_tool(ticker: str):
    """Données financières (Prix, PER, Dividende)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # On ne garde que l'essentiel strict pour économiser des tokens
        data = {
            "Nom": info.get('longName'),
            "Prix": info.get('currentPrice'),
            "PER": info.get('forwardPE'),
            "Div": (info.get('dividendYield', 0) or 0) * 100,
            "Avis": info.get('recommendationKey')
        }
        return str(data)
    except Exception:
        return "Erreur données."

# --- 6. MOTEUR DE L'AGENT ---
def run_crew(ticker_symbol):
    os.environ["GROQ_API_KEY"] = api_key
    
    # Modèle Rapide 8B
    my_llm = LLM(
        model="groq/llama-3.1-8b-instant",
        temperature=0.1
    )

    # Agents (Descriptions raccourcies pour économiser le quota)
    analyste = Agent(
        role='Analyste',
        goal='Donner les chiffres clés',
        backstory="Expert comptable factuel.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[analyse_bourse_tool],
        max_rpm=10 # Limite la vitesse pour éviter l'erreur 429
    )

    trader = Agent(
        role='Trader',
        goal='Donner le sentiment web',
        backstory="Expert réseaux sociaux.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[recherche_web_tool],
        max_rpm=10 # Limite la vitesse
    )

    # Tâches
    task_finance = Task(
        description=f"Donne Prix, PER, Dividende pour {ticker_symbol}.",
        expected_output="Chiffres clés.",
        agent=analyste
    )

    task_sentiment = Task(
        description=f"Cherche avis sur {ticker_symbol} (Twitter/Reddit).",
        expected_output="Sentiment global (Bref).",
        agent=trader
    )

    task_synthese = Task(
        description=f"Synthèse pour {ticker_symbol} (PEA). Achat/Vente ? Court.",
        expected_output="Conseil final court.",
        agent=analyste,
        context=[task_finance, task_sentiment]
    )

    crew = Crew(
        agents=[analyste, trader],
        tasks=[task_finance, task_sentiment, task_synthese],
        process=Process.sequential,
        memory=False,
        verbose=True
    )

    return crew.kickoff()

# --- 7. EXÉCUTION AVEC RETRY ---
ticker_input = st.text_input("Action (ex: TTE.PA)", "TTE.PA")

if st.button("Lancer 🚀"):
    if not api_key:
        st.error("Clé manquante !")
    else:
        with st.status("Analyse en cours...", expanded=True) as status:
            try:
                st.write("🔄 Démarrage (Si ça prend du temps, c'est la pause anti-spam)...")
                
                # Système de Retry manuel
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        resultat = run_crew(ticker_input)
                        status.update(label="✅ Terminé !", state="complete", expanded=False)
                        st.markdown("### 📊 Résultat")
                        st.markdown(resultat)
                        break # Si ça marche, on sort de la boucle
                    except Exception as e:
                        if "429" in str(e) or "Rate limit" in str(e):
                            wait_time = 10 * (attempt + 1)
                            st.write(f"⚠️ Limite API atteinte. Pause de {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            raise e # Si c'est une autre erreur, on plante vraiment
                            
            except Exception as e:
                st.error(f"Erreur finale : {e}")
                status.update(label="Erreur", state="error")