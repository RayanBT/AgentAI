import streamlit as st
import os
import time
import re
import pandas as pd
import google.generativeai as genai
import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# --- 1. CONFIGURATION ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

st.set_page_config(page_title="Agent PEA Hunter", page_icon="🎯", layout="wide")

st.title("🎯 Assistant PEA (Chasseur d'Opportunités)")
st.markdown("Mode **Chasseur Actif** : L'IA identifie les tendances du moment et note le potentiel des actions.")

# --- 2. STRATÉGIES DE CHASSE ---
HUNTING_STRATEGIES = {
    "💎 Pépites Cachées (Small Caps)": "Trouve 5 actions françaises (PEA) de petite ou moyenne capitalisation qui sont actuellement considérées comme sous-évaluées ou en retournement. Cherche des noms moins connus que le CAC40.",
    "🚀 Fort Potentiel de Croissance": "Trouve 5 actions européennes éligibles PEA dans les secteurs technologiques, énergies vertes ou santé qui ont une forte dynamique de croissance récente.",
    "🛡️ Rendement & Sécurité (Défensif)": "Trouve 5 actions françaises solides (hors top 10 CAC40) qui offrent un dividende stable et sont peu volatiles en ce moment.",
    "🔥 Momentum (Ça buzz en ce moment)": "Trouve 5 actions éligibles PEA qui font l'actualité en ce moment (fusions, résultats exceptionnels, contrats) et dont on parle sur les forums boursiers."
}

# --- 3. FONCTIONS UTILITAIRES ---
def get_active_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        valid = [m.name for m in models if 'generateContent' in m.supported_generation_methods and not any(x in m.name for x in ['tts', 'vision', 'embedding'])]
        return sorted(valid, key=lambda x: 0 if "gemini-2.0-flash" in x else 1)
    except: return []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé Google API", type="password")
    if api_key:
        models = get_active_models(api_key)
        if models:
            crew_models = [m.replace("models/", "gemini/") for m in models]
            st.success(f"✅ Système armé ({len(models)} modèles)")
        else:
            st.error("Aucun modèle valide.")
            crew_models = []

# --- 5. OUTILS ---
@tool("Recherche Web")
def recherche_web_tool(query: str):
    """Recherche Web."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
            return "\n".join([f"- {r['body']}" for r in results]) if results else "Rien."
    except: return "Erreur."

@tool("Bourse Yahoo")
def analyse_bourse_tool(ticker: str):
    """Données."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return str({
            "Prix": info.get('currentPrice'),
            "PER": info.get('forwardPE'),
            "Div": info.get('dividendYield')
        })
    except: return "Erreur Yahoo."

# --- 6. MOTEUR ROBUSTE (Identique) ---
def execute_step_smart(step_name, task_desc, role, tools, model_list, context=""):
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key
    
    max_retries = 3
    current_retry = 0
    
    while current_retry <= max_retries:
        retry_delays = []
        for model_name in model_list:
            try:
                my_llm = LLM(model=model_name, api_key=api_key, temperature=0.1)
                agent = Agent(role=role, goal="Tâche", backstory="Expert.", verbose=True, allow_delegation=False, llm=my_llm, tools=tools, max_rpm=10)
                desc = task_desc + (f"\nCONTEXTE:\n{context}" if context else "")
                task = Task(description=desc, expected_output="Court.", agent=agent)
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                return str(crew.kickoff())
            except Exception as e:
                err = str(e)
                if "429" in err or "Quota" in err:
                    match = re.search(r"retry in (\d+\.?\d*)s", err)
                    wait = float(match.group(1)) if match else 60.0
                    retry_delays.append(wait)
                    time.sleep(1)
                    continue
                if "404" in err or "400" in err: continue
        
        if not retry_delays: return None
        if current_retry < max_retries:
            wait_time = min(retry_delays) + 2
            with st.empty():
                for i in range(int(wait_time), 0, -1):
                    st.toast(f"🛑 Pause Quota... Reprise dans {i}s", icon="⏳")
                    time.sleep(1)
            current_retry += 1
        else: return None

# --- 7. PHASE DE CHASSE (NOUVEAU) ---
def hunt_tickers(strategy_prompt):
    """Utilise l'IA pour générer une liste de tickers basés sur la stratégie."""
    if not crew_models: return []
    
    prompt = f"""
    Tu es un expert en marché boursier européen.
    Ta mission : {strategy_prompt}
    
    IMPORTANT :
    1. Donne-moi UNIQUEMENT une liste de 4 à 5 symboles boursiers (Tickers).
    2. Les symboles doivent être au format Yahoo Finance (ex: "AIR.PA" pour Airbus, "AI.PA" pour Air Liquide).
    3. Choisis des actions réelles et liquides.
    4. Ne mets pas de texte, juste les tickers séparés par des virgules.
    Exemple de réponse attendue : TTE.PA, MC.PA, GLE.PA, RNO.PA
    """
    
    # On utilise un "One-Shot Agent" pour cette tâche rapide
    res = execute_step_smart("Stratège", prompt, "Stratège de Marché", [recherche_web_tool], crew_models)
    
    if res:
        # Nettoyage de la réponse pour extraire les tickers
        raw_tickers = res.replace(" ", "").replace("\n", "").split(",")
        # Filtrage basique pour garder le format X.PA ou X.DE
        clean_tickers = [t for t in raw_tickers if "." in t]
        return clean_tickers[:5] # On limite à 5 pour le quota
    return []

# --- 8. ANALYSE PROFONDE AVEC SCORING ---
def analyze_one_stock(ticker, progress_callback=None):
    if not crew_models: return None
    
    dossier = ""
    if progress_callback: progress_callback(f"🔎 {ticker} : Analyse Finance...")
    res_fin = execute_step_smart("Finance", f"Prix, PER, Div de {ticker}", "Analyste", [analyse_bourse_tool], crew_models)
    if not res_fin: return None
    dossier += f"FINANCE: {res_fin}\n"
    
    if progress_callback: progress_callback(f"🔎 {ticker} : Analyse Sentiment...")
    res_soc = execute_step_smart("Sentiment", f"Avis web sur {ticker}", "Trader", [recherche_web_tool], crew_models)
    if not res_soc: return None
    dossier += f"SENTIMENT: {res_soc}\n"
    
    if progress_callback: progress_callback(f"🔎 {ticker} : Notation...")
    # Demande spécifique pour avoir un score numérique
    res_con = execute_step_smart(
        "Notation", 
        f"Analyse le dossier pour {ticker}. Donne une note de potentiel sur 10 (ex: 8/10) puis une recommandation (ACHAT/VENTE) et une phrase d'explication.", 
        "Conseiller", [], crew_models, dossier
    )
    
    # Extraction de la note (Regex simple)
    score = 0
    match = re.search(r"(\d+)/10", str(res_con))
    if match:
        score = int(match.group(1))
    
    return {
        "Action": ticker,
        "Score": score,
        "Note": f"{score}/10",
        "Avis": res_con
    }

# --- 9. INTERFACE ---
tab1, tab2 = st.tabs(["🔍 Analyse Solo", "🎯 Chasseur de Pépites"])

# --- TAB 1 : SOLO ---
with tab1:
    ticker_input = st.text_input("Action", "TTE.PA")
    if st.button("Lancer Solo 🚀"):
        if not api_key: st.error("Clé manquante")
        else:
            with st.status("Analyse en cours...", expanded=True):
                res = analyze_one_stock(ticker_input, st.write)
                if res:
                    st.success("Terminé !")
                    st.markdown(f"### Note : {res['Note']}")
                    st.write(res['Avis'])

# --- TAB 2 : CHASSEUR ---
with tab2:
    st.header("Définir la stratégie de chasse")
    
    strategy_name = st.selectbox("Quel type d'opportunité cherches-tu ?", list(HUNTING_STRATEGIES.keys()))
    
    if st.button("Lancer la Chasse 🦅"):
        if not api_key:
            st.error("Clé manquante")
        else:
            # 1. PHASE DE RECHERCHE
            with st.status("📡 Le Stratège scanne le marché...", expanded=True) as status:
                st.write("🧠 Analyse des tendances en cours...")
                tickers_found = hunt_tickers(HUNTING_STRATEGIES[strategy_name])
                
                if not tickers_found:
                    status.update(label="Aucune cible trouvée.", state="error")
                    st.error("Le stratège n'a pas trouvé d'actions correspondant aux critères.")
                else:
                    st.success(f"Cibles identifiées : {', '.join(tickers_found)}")
                    status.update(label="Cibles verrouillées. Début de l'analyse profonde...", state="running")
                    
                    # 2. PHASE D'ANALYSE
                    results_data = []
                    progress_bar = st.progress(0)
                    table_placeholder = st.empty()
                    
                    total = len(tickers_found)
                    
                    for i, ticker in enumerate(tickers_found):
                        res = analyze_one_stock(ticker, st.write)
                        
                        if res:
                            results_data.append(res)
                            # Mise à jour tableau
                            df = pd.DataFrame(results_data)
                            # Tri par Score décroissant
                            df = df.sort_values(by="Score", ascending=False)
                            
                            table_placeholder.dataframe(
                                df[["Action", "Note", "Avis"]], 
                                use_container_width=True,
                                column_config={
                                    "Avis": st.column_config.TextColumn("Analyse IA", width="large")
                                }
                            )
                        
                        progress_bar.progress((i + 1) / total)
                        if i < total - 1: time.sleep(2)

                    status.update(label="Chasse terminée !", state="complete", expanded=False)
                    st.balloons()