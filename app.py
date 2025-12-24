import streamlit as st
import os
import time
import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# --- 1. CONFIGURATION SYSTEME ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

# --- 2. INTERFACE ---
st.set_page_config(page_title="Agent PEA Ultra-Free", page_icon="💰")
st.title("💰 Assistant PEA (Mode Gratuit Optimisé)")
st.markdown("Analyse financière & Sentiment - **Optimisé pour les quotas Google**")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Ta clé Google API (AIza...)", type="password")
    if not api_key:
        st.warning("Entre ta clé.")
    else:
        st.success("Clé prête !")

# --- 4. OUTILS (SIMPLIFIÉS POUR ÉCONOMISER DES TOKENS) ---

@tool("Recherche Web")
def recherche_web_tool(query: str):
    """Cherche sur le web. Utiliser des mots-clés précis."""
    try:
        # On ne prend que 2 résultats pour économiser la bande passante du modèle
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
            if not results: return "Rien trouvé."
            return "\n".join([f"- {r['body']}" for r in results])
    except: return "Erreur recherche."

@tool("Bourse Yahoo")
def analyse_bourse_tool(ticker: str):
    """Données financières basiques."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # On ne garde que l'essentiel strict
        return str({
            "Prix": info.get('currentPrice'),
            "PER": info.get('forwardPE'),
            "Div(%)": (info.get('dividendYield', 0) or 0) * 100
        })
    except: return "Erreur Yahoo."

# --- 5. MOTEUR ---
def run_crew(ticker_symbol):
    
    # Injection des clés
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # --- STRATÉGIE 1 : LE BON MODÈLE ---
    # gemini-1.5-flash est LE modèle gratuit avec les plus hauts quotas (15 RPM)
    # On évite le '2.0' ou 'pro' qui saturent vite.
    my_llm = LLM(
        model="gemini/gemini-1.5-flash", 
        api_key=api_key,
        temperature=0.1 # Très factuel pour éviter le blabla inutile
    )

    # --- STRATÉGIE 2 : FREINS AUX AGENTS ---
    # max_rpm=5 : L'agent ne fera pas plus de 5 appels par minute.
    # Cela force une pause naturelle et évite l'erreur 429.
    
    analyste = Agent(
        role='Analyste',
        goal='Donner chiffres',
        backstory="Expert bref.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[analyse_bourse_tool],
        max_rpm=5 # <--- LE FREIN
    )

    trader = Agent(
        role='Trader',
        goal='Donner sentiment',
        backstory="Expert bref.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm,
        tools=[recherche_web_tool],
        max_rpm=5 # <--- LE FREIN
    )

    # Tâches (Courtes pour économiser des tokens)
    task1 = Task(
        description=f"Donne Prix, PER, Dividende pour {ticker_symbol}.",
        expected_output="Chiffres simples.",
        agent=analyste
    )

    task2 = Task(
        description=f"Cherche avis web sur {ticker_symbol}.",
        expected_output="Sentiment (Positif/Négatif).",
        agent=trader
    )

    task3 = Task(
        description=f"Conclusion PEA pour {ticker_symbol} (Achat/Attente). Court.",
        expected_output="Conseil final en 3 phrases max.",
        agent=analyste,
        context=[task1, task2]
    )

    # Crew
    crew = Crew(
        agents=[analyste, trader],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        memory=False,
        verbose=True
    )

    return crew.kickoff()

# --- 6. EXÉCUTION ---
ticker = st.text_input("Action (ex: TTE.PA)", "TTE.PA")

if st.button("Lancer l'analyse 🚀"):
    if not api_key:
        st.error("Clé manquante !")
    else:
        with st.status("Analyse en cours (Mode Prudent)...", expanded=True) as status:
            try:
                # Petite pause initiale pour laisser le système respirer
                time.sleep(2)
                st.write("🐢 Démarrage en mode régulé...")
                
                resultat = run_crew(ticker)
                
                status.update(label="✅ Réussi !", state="complete", expanded=False)
                st.divider()
                st.markdown("### 📊 Résultat")
                st.markdown(resultat)
                
            except Exception as e:
                # Gestion propre de l'erreur de quota
                if "429" in str(e):
                    st.warning("⚠️ Limite de vitesse Google atteinte. Attends 1 minute et réessaie.")
                    st.error(f"Détail technique : {e}")
                else:
                    st.error(f"Erreur : {e}")