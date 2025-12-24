import streamlit as st
import os
import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# --- 1. CONFIGURATION SYSTÈME (ANTI-CRASH) ---
# On coupe la télémétrie
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
# On donne une fausse clé OpenAI pour satisfaire les vérifications initiales
os.environ["OPENAI_API_KEY"] = "NA"

# --- 2. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Agent PEA Gemini", page_icon="💎", layout="wide")

st.title("💎 Assistant PEA (Gemini Flash Native)")
st.markdown("""
Cet agent utilise **Gemini 1.5 Flash** via le connecteur natif.
C'est la méthode la plus fiable pour éviter les erreurs OpenAI.
""")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Ta clé Google API (AIza...)", type="password")
    if not api_key:
        st.warning("Entre ta clé pour démarrer.")
        st.markdown("[Obtenir une clé Google ici](https://aistudio.google.com/app/apikey)")

# --- 4. DÉFINITION DES OUTILS (STABLES) ---

@tool("Recherche Web")
def recherche_web_tool(query: str):
    """
    Recherche sur internet (News, Sentiment).
    """
    try:
        # Utilisation directe de la librairie pour éviter les bugs d'import CrewAI
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
            if not results:
                return "Pas de résultat."
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Erreur recherche : {e}"

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
            "PER": info.get('forwardPE', 'N/A'),
            "Dividende (%)": (info.get('dividendYield', 0) or 0) * 100,
            "Recommandation": info.get('recommendationKey', 'Inconnue')
        }
        return str(data)
    except Exception as e:
        return f"Erreur Yahoo : {e}"

# --- 5. MOTEUR DE L'AGENT ---
def run_crew(ticker_symbol):
    
    # On force la clé dans l'environnement système aussi (ceinture et bretelles)
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # --- CERVEAU GEMINI NATIF ---
    # La syntaxe "gemini/..." dit à CrewAI : "Utilise Google, pas OpenAI !"
    my_llm = LLM(
        model="gemini/gemini-1.5-flash",
        api_key=api_key,
        temperature=0.3
    )

    # Agent 1 : Financier
    analyste = Agent(
        role='Analyste Financier',
        goal='Analyser les chiffres clés',
        backstory="Expert comptable rigoureux.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm, # On connecte le cerveau
        tools=[analyse_bourse_tool]
    )

    # Agent 2 : Social
    trader = Agent(
        role='Expert Sentiment',
        goal='Sonder le web',
        backstory="Expert réseaux sociaux.",
        verbose=True,
        allow_delegation=False,
        llm=my_llm, # On connecte le cerveau
        tools=[recherche_web_tool]
    )

    # Tâches
    task_finance = Task(
        description=f"Donne les chiffres (Prix, PER, Dividende) de {ticker_symbol}.",
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
        memory=False, # Toujours désactivé pour la vitesse
        verbose=True
    )

    return crew.kickoff()

# --- 6. EXÉCUTION ---
ticker_input = st.text_input("Symbole de l'action (ex: TTE.PA)", "TTE.PA")

if st.button("Lancer l'Analyse 🚀"):
    if not api_key:
        st.error("⚠️ Clé manquante !")
    else:
        with st.status("Gemini s'active...", expanded=True) as status:
            try:
                st.write("💎 Initialisation...")
                resultat = run_crew(ticker_input)
                status.update(label="✅ Terminé !", state="complete", expanded=False)
                
                st.divider()
                st.markdown("### 📊 Rapport Final")
                st.markdown(resultat)
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                status.update(label="Erreur", state="error")