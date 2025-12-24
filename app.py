import streamlit as st
import os
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from duckduckgo_search import DDGS

# --- 1. CONFIGURATION SYSTÈME ---
# On coupe la télémétrie et on fait taire les warnings OpenAI
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

# --- 2. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Agent PEA Gemini", page_icon="💎", layout="wide")

st.title("💎 Assistant PEA (Propulsé par Google Gemini)")
st.markdown("""
Cet agent utilise **Gemini 1.5 Flash**.  
Il est plus stable, plus rapide et a une limite d'utilisation beaucoup plus large que Groq.
""")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Ta clé Google API", type="password")
    if not api_key:
        st.warning("Entre ta clé pour démarrer.")
        st.markdown("[Obtenir une clé Google ici](https://aistudio.google.com/app/apikey)")

# --- 4. DÉFINITION DES OUTILS (ROBUSTES) ---

@tool("Recherche Web")
def recherche_web_tool(query: str):
    """
    Recherche sur internet (X, Reddit, News).
    """
    try:
        # On utilise DDGS directement pour éviter les bugs de dépendances
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "Aucun résultat trouvé sur le web."
            # On formate proprement
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Erreur de recherche : {e}"

@tool("Bourse Yahoo")
def analyse_bourse_tool(ticker: str):
    """
    Récupère les données financières (Prix, PER, Dividende).
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Sécurisation des données si elles manquent
        data = {
            "Nom": info.get('longName', ticker),
            "Prix": info.get('currentPrice', 'N/A'),
            "Devise": info.get('currency', 'EUR'),
            "PER (Price/Earnings)": info.get('forwardPE', 'N/A'),
            "Dividende (%)": (info.get('dividendYield', 0) or 0) * 100,
            "Recommandation Analystes": info.get('recommendationKey', 'Inconnue'),
            "Secteur": info.get('sector', 'N/A')
        }
        return str(data)
    except Exception as e:
        return f"Erreur Yahoo Finance : {e}"

# --- 5. MOTEUR DE L'AGENT ---
def run_crew(ticker_symbol):
    
    # --- LE CERVEAU GOOGLE ---
    # On configure Gemini via LangChain (très stable avec CrewAI)
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        verbose=True,
        temperature=0.4,
        google_api_key=api_key
    )

    # Agent 1 : L'Analyste Financier
    analyste = Agent(
        role='Analyste Financier Senior',
        goal='Analyser les fondamentaux boursiers',
        backstory="Tu es un expert comptable rigoureux. Tu ne te fies qu'aux chiffres (Dividendes, PER, Croissance).",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[analyse_bourse_tool]
    )

    # Agent 2 : L'Expert Réseaux Sociaux
    trader = Agent(
        role='Expert Sentiment Marché',
        goal='Sonder l\'opinion publique sur le web',
        backstory="Tu es un trader connecté qui scanne le web pour sentir la psychologie des investisseurs (Peur ou Euphorie).",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[recherche_web_tool]
    )

    # Tâches
    task_finance = Task(
        description=f"Analyse les fondamentaux de l'action {ticker_symbol}. Je veux le Prix, le PER et le Dividende.",
        expected_output="Un résumé des données financières clés.",
        agent=analyste
    )

    task_sentiment = Task(
        description=f"Cherche sur le web ce qu'on dit de {ticker_symbol}. Regarde les avis récents.",
        expected_output="Une synthèse de l'ambiance (Positive/Négative/Neutre).",
        agent=trader
    )

    task_synthese = Task(
        description=f"En combinant les chiffres et l'ambiance, rédige une recommandation pour un investisseur PEA sur {ticker_symbol}.",
        expected_output="Un rapport final clair avec une recommandation (Achat/Vente/Attente) justifiée.",
        agent=analyste,
        context=[task_finance, task_sentiment]
    )

    # Lancement de l'équipe
    crew = Crew(
        agents=[analyste, trader],
        tasks=[task_finance, task_sentiment, task_synthese],
        process=Process.sequential,
        memory=False, # On garde la mémoire désactivée pour la vitesse
        verbose=True
    )

    return crew.kickoff()

# --- 6. EXÉCUTION ---
ticker_input = st.text_input("Symbole de l'action (ex: TTE.PA, AI.PA)", "TTE.PA")

if st.button("Lancer l'Analyse avec Gemini 🚀"):
    if not api_key:
        st.error("⚠️ Il manque ta clé Google API dans la barre latérale !")
    else:
        with st.status("Les agents Gemini travaillent...", expanded=True) as status:
            try:
                st.write("🌍 Initialisation de l'équipe...")
                resultat = run_crew(ticker_input)
                status.update(label="✅ Analyse Terminée !", state="complete", expanded=False)
                
                st.divider()
                st.markdown("### 📊 Rapport Final")
                st.markdown(resultat)
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                status.update(label="Erreur", state="error")