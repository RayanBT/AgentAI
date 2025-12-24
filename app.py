import streamlit as st
import os

# --- 1. CONFIGURATION SYSTÈME (Pour la stabilité Streamlit) ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# --- 2. IMPORTS ---
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

# IMPORTS STABLES LANGCHAIN (C'est la clé de la réussite)
# On importe le décorateur @tool depuis LangChain Core
from langchain.tools import tool
# On importe le moteur de recherche depuis LangChain Community
from langchain_community.tools import DuckDuckGoSearchRun

# --- 3. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Agent PEA Intelligent", page_icon="📈")
st.title("📈 Assistant PEA Intelligent")
st.markdown("Analyse financière & Sentiment social (X/Reddit) - **Architecture Robuste**")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Ta clé API Groq", type="password")
    if not api_key:
        st.warning("Entre ta clé pour démarrer.")
        st.info("Obtenir une clé : https://console.groq.com")

# --- 5. DÉFINITION DES OUTILS (Via LangChain) ---

# Outil 1 : Recherche Web
@tool
def recherche_web_tool(query: str):
    """
    Utilise cet outil pour faire des recherches sur Internet.
    Utile pour trouver des actualités récentes, des avis sur des forums (Reddit)
    ou des discussions sur les réseaux sociaux (X/Twitter).
    """
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Erreur lors de la recherche : {e}"

# Outil 2 : Analyse Bourse
@tool
def analyse_bourse_tool(ticker: str):
    """
    Récupère les données financières d'une action via son ticker Yahoo Finance.
    Exemple de ticker : 'TTE.PA' pour TotalEnergies, 'AI.PA' pour Air Liquide.
    Renvoie le prix, le PER, le dividende et la recommandation des analystes.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # On sécurise la récupération des données (parfois nulles)
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
        return f"Erreur lors de la récupération des données boursières pour {ticker} : {e}"

# --- 6. MOTEUR DE L'AGENT ---
def run_crew(ticker_symbol):
    # Initialisation du LLM (Cerveau)
    llm = ChatGroq(
        api_key=api_key,
        model="llama3-70b-8192",
        temperature=0.5
    )

    # Agent 1 : L'Analyste Chiffré
    analyste = Agent(
        role='Analyste Financier Senior',
        goal='Évaluer la santé financière et la rentabilité d\'une action',
        backstory="Tu es un expert-comptable rigoureux. Tu ne crois que les chiffres (Bilan, Dividendes, PER).",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[analyse_bourse_tool] # Utilise l'outil LangChain
    )

    # Agent 2 : Le Trader Sentiment (Réseaux Sociaux)
    trader = Agent(
        role='Analyste Sentiment de Marché',
        goal='Sonder l\'opinion publique sur X (Twitter) et Reddit',
        backstory="Tu es un trader connecté H24. Tu cherches les rumeurs, le FOMO ou la panique sur les réseaux.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[recherche_web_tool] # Utilise l'outil LangChain
    )

    # Tâches
    task_finance = Task(
        description=f"Analyse les fondamentaux de {ticker_symbol}. Cherche le prix, le dividende et le PER.",
        expected_output="Un résumé des chiffres clés.",
        agent=analyste
    )

    task_sentiment = Task(
        description=f"""
        Va chercher sur le web ce que les gens disent de {ticker_symbol}.
        Utilise des requêtes comme 'site:twitter.com {ticker_symbol} avis' ou 'site:reddit.com {ticker_symbol} PEA'.
        Est-ce que l'ambiance est positive ou négative ?
        """,
        expected_output="Une analyse de l'humeur du marché.",
        agent=trader
    )

    task_synthese = Task(
        description=f"""
        En utilisant les rapports financiers et sociaux, rédige une recommandation finale pour {ticker_symbol}.
        Dois-je l'intégrer dans mon PEA ? (Achat / Vente / Attente).
        Justifie ta réponse.
        """,
        expected_output="Un rapport d'investissement complet en Français au format Markdown.",
        agent=analyste,
        context=[task_finance, task_sentiment]
    )

    # Lancement de l'équipe
    crew = Crew(
        agents=[analyste, trader],
        tasks=[task_finance, task_sentiment, task_synthese],
        process=Process.sequential
    )

    return crew.kickoff()

# --- 7. EXÉCUTION ---
ticker_input = st.text_input("Symbole de l'action (ex: TTE.PA, MC.PA)", "TTE.PA")

if st.button("Lancer l'Analyse 🚀"):
    if not api_key:
        st.error("⚠️ Il manque la clé API Groq dans la colonne de gauche !")
    else:
        with st.status("Les agents IA travaillent...", expanded=True) as status:
            try:
                st.write("🔄 Initialisation des agents...")
                resultat = run_crew(ticker_input)
                
                status.update(label="✅ Analyse Terminée !", state="complete", expanded=False)
                
                st.divider()
                st.markdown("### 📊 Rapport Final pour ton PEA")
                st.markdown(resultat)
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                status.update(label="❌ Erreur", state="error")