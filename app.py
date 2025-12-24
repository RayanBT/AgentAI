import streamlit as st
import os
import time
import google.generativeai as genai  # Pour scanner les modèles
import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# --- 1. CONFIGURATION SYSTEME ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

# --- 2. INTERFACE ---
st.set_page_config(page_title="Agent PEA Sélecteur", page_icon="🎛️", layout="wide")
st.title("🎛️ Assistant PEA (Sélecteur de Modèle)")
st.markdown("Analyse financière & Sentiment - **Choisis ton modèle Gemini**")

# --- 3. SIDEBAR INTELLIGENTE ---
with st.sidebar:
    st.header("1. Configuration")
    api_key = st.text_input("Ta clé Google API", type="password")
    
    selected_model_string = None
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GEMINI_API_KEY"] = api_key
        
        try:
            # On demande à Google : "Quels modèles j'ai le droit d'utiliser ?"
            genai.configure(api_key=api_key)
            models = list(genai.list_models())
            
            # On garde ceux qui savent écrire du texte
            valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            
            if valid_models:
                st.success(f"{len(valid_models)} modèles trouvés !")
                # L'utilisateur choisit ici
                raw_model_name = st.selectbox(
                    "2. Choisis un modèle (Prends un 'Flash' ou 'Pro')", 
                    valid_models,
                    index=0
                )
                # Nettoyage du nom pour CrewAI (on enlève 'models/')
                selected_model_string = raw_model_name.replace("models/", "")
                st.info(f"Modèle actif : {selected_model_string}")
            else:
                st.error("Aucun modèle compatible trouvé.")
                
        except Exception as e:
            st.error(f"Erreur de clé : {e}")

# --- 4. OUTILS ---
@tool("Recherche Web")
def recherche_web_tool(query: str):
    """Recherche Web."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
            if not results: return "Rien trouvé."
            return "\n".join([f"- {r['body']}" for r in results])
    except: return "Erreur recherche."

@tool("Bourse Yahoo")
def analyse_bourse_tool(ticker: str):
    """Données financières."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return str({
            "Prix": info.get('currentPrice'),
            "PER": info.get('forwardPE'),
            "Div": info.get('dividendYield')
        })
    except: return "Erreur Yahoo."

# --- 5. MOTEUR ---
def run_crew(ticker, model_name):
    
    # On construit le nom technique pour CrewAI (ex: gemini/gemini-1.5-flash-001)
    llm_model_name = f"gemini/{model_name}"
    
    my_llm = LLM(
        model=llm_model_name,
        api_key=api_key,
        temperature=0.1
    )

    # Agents (Freinés à 5 RPM pour éviter le quota error)
    analyste = Agent(
        role='Analyste', goal='Chiffres', backstory="Expert.",
        verbose=True, allow_delegation=False, llm=my_llm, tools=[analyse_bourse_tool],
        max_rpm=5
    )

    trader = Agent(
        role='Trader', goal='Sentiment', backstory="Expert.",
        verbose=True, allow_delegation=False, llm=my_llm, tools=[recherche_web_tool],
        max_rpm=5
    )

    task1 = Task(description=f"Chiffres {ticker}.", expected_output="Données.", agent=analyste)
    task2 = Task(description=f"Avis web {ticker}.", expected_output="Avis.", agent=trader)
    task3 = Task(description=f"Synthèse PEA {ticker}.", expected_output="Conseil.", agent=analyste, context=[task1, task2])

    crew = Crew(agents=[analyste, trader], tasks=[task1, task2, task3], process=Process.sequential, verbose=True, memory=False)
    return crew.kickoff()

# --- 6. EXÉCUTION ---
if selected_model_string:
    st.divider()
    ticker = st.text_input("Action", "TTE.PA")
    
    if st.button("Lancer l'analyse 🚀"):
        with st.status("Travail en cours...", expanded=True) as status:
            try:
                st.write(f"🤖 Utilisation de : {selected_model_string}")
                time.sleep(1) # Petite pause sécurité
                
                resultat = run_crew(ticker, selected_model_string)
                
                status.update(label="Terminé !", state="complete")
                st.divider()
                st.markdown(resultat)
            except Exception as e:
                st.error(f"Erreur : {e}")