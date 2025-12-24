import streamlit as st
import os
import google.generativeai as genai
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS
import yfinance as yf

# --- CONFIGURATION ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

st.set_page_config(page_title="Diagnostic Gemini", page_icon="🔧")

st.title("🔧 Diagnostic & Assistant PEA")
st.markdown("Ce mode permet de trouver le bon modèle Gemini compatible avec ta clé.")

# --- SIDEBAR & DIAGNOSTIC ---
with st.sidebar:
    st.header("1. Clé API")
    api_key = st.text_input("Ta clé Google AIza...", type="password")
    
    selected_model_name = None
    
    if api_key:
        try:
            # On configure le SDK Google directement
            genai.configure(api_key=api_key)
            
            # On demande la liste des modèles
            st.success("Clé détectée ! Recherche des modèles...")
            models = list(genai.list_models())
            
            # On filtre pour ne garder que ceux qui génèrent du texte (gemini)
            gemini_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            
            if gemini_models:
                st.header("2. Choisis ton Modèle")
                # L'utilisateur choisit le modèle dans la liste réelle
                selected_model_name = st.selectbox(
                    "Modèles disponibles pour ta clé :", 
                    gemini_models,
                    index=0
                )
                # On nettoie le nom (Google renvoie 'models/gemini-pro', CrewAI veut 'gemini/gemini-pro')
                clean_name = selected_model_name.replace("models/", "")
                st.info(f"Modèle sélectionné : {clean_name}")
            else:
                st.error("Aucun modèle Gemini trouvé pour cette clé.")
                
        except Exception as e:
            st.error(f"Erreur de connexion Google : {e}")

# --- OUTILS ---
@tool("Recherche Web")
def recherche_web_tool(query: str):
    """Recherche Web."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results: return "Rien trouvé."
            return "\n".join([f"- {r['body']}" for r in results])
    except: return "Erreur recherche."

@tool("Bourse Yahoo")
def analyse_bourse_tool(ticker: str):
    """Données Bourse."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return str({
            "Nom": info.get('longName'),
            "Prix": info.get('currentPrice'),
            "PER": info.get('forwardPE'),
            "Div": info.get('dividendYield')
        })
    except: return "Erreur Yahoo."

# --- MOTEUR ---
def run_crew(ticker, model_name):
    # Injection des clés
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # On utilise le modèle choisi dynamiquement
    # Note : CrewAI attend 'gemini/nom-du-modele'
    full_model_name = f"gemini/{model_name}"
    
    my_llm = LLM(
        model=full_model_name,
        api_key=api_key,
        temperature=0.3
    )

    analyste = Agent(
        role='Analyste', goal='Chiffres', backstory='Expert.',
        allow_delegation=False, verbose=True, llm=my_llm, tools=[analyse_bourse_tool]
    )
    trader = Agent(
        role='Trader', goal='Sentiment', backstory='Expert.',
        allow_delegation=False, verbose=True, llm=my_llm, tools=[recherche_web_tool]
    )
    
    task1 = Task(description=f"Donne les chiffres de {ticker}.", expected_output="Chiffres.", agent=analyste)
    task2 = Task(description=f"Donne le sentiment sur {ticker}.", expected_output="Sentiment.", agent=trader)
    task3 = Task(description=f"Avis final PEA pour {ticker}.", expected_output="Avis.", agent=analyste, context=[task1, task2])

    crew = Crew(agents=[analyste, trader], tasks=[task1, task2, task3], process=Process.sequential, verbose=True)
    return crew.kickoff()

# --- MAIN ---
if selected_model_name:
    st.divider()
    ticker = st.text_input("Action", "TTE.PA")
    if st.button("Lancer l'analyse"):
        with st.status("Travail en cours..."):
            try:
                # On passe le nom nettoyé (ex: gemini-1.5-flash)
                clean_name = selected_model_name.replace("models/", "")
                res = run_crew(ticker, clean_name)
                st.markdown(res)
            except Exception as e:
                st.error(f"Erreur : {e}")