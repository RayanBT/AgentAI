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

st.set_page_config(page_title="Agent PEA Transparent", page_icon="🧿", layout="wide")

st.title("🧿 Assistant PEA (Mode Transparent)")
st.markdown("Suivez la réflexion des agents IA étape par étape en temps réel.")

# --- 2. STRATÉGIES ---
HUNTING_STRATEGIES = {
    "💎 Pépites Cachées (Small Caps)": "Trouve 4 actions françaises (PEA) de petite/moyenne capitalisation sous-évaluées. Cherche hors du CAC40.",
    "🚀 Croissance Tech/Verte": "Trouve 4 actions européennes (PEA) Tech ou Green Energy avec forte croissance.",
    "🛡️ Rendement & Dividende": "Trouve 4 actions françaises solides avec un haut rendement de dividende (>5%) et stable.",
    "🔥 Momentum (Buzz actuel)": "Trouve 4 actions PEA qui font l'actualité positivement cette semaine."
}

# --- 3. FONCTIONS UTILITAIRES ---
@st.cache_data(show_spinner=False)
def get_active_models(api_key):
    """Scan les modèles (Mis en cache pour la vitesse)."""
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        valid_models = []
        for m in models:
            name = m.name.lower()
            if 'generateContent' not in m.supported_generation_methods: continue
            if any(x in name for x in ['tts', 'vision', 'embedding', 'geek', 'gecko']): continue
            valid_models.append(m.name)
        
        return sorted(valid_models, key=lambda x: (
            0 if "gemini-1.5-flash" in x and "8b" not in x else 1 if "gemini-2.0-flash" in x else 2
        ))
    except: return []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé Google API", type="password")
    if api_key:
        models = get_active_models(api_key)
        if models:
            crew_models = [m.replace("models/", "gemini/") for m in models]
            st.success(f"✅ {len(models)} cerveaux connectés")
        else: st.error("Aucun modèle.")
    else: crew_models = []

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

# --- 6. MOTEUR BAVARD (VERBOSE ENGINE) ---
def execute_step_smart(step_name, task_desc, role, tools, model_list, log_container, context=""):
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key
    
    max_retries = 3
    current_retry = 0
    
    while current_retry <= max_retries:
        retry_delays = []
        for model_name in model_list:
            clean_name = model_name.replace("gemini/", "")
            
            try:
                # Log visuel
                log_container.markdown(f"⚡ Tentative avec **{clean_name}**...")
                
                my_llm = LLM(model=model_name, api_key=api_key, temperature=0.1)
                agent = Agent(role=role, goal="Tâche", backstory="Expert.", verbose=True, allow_delegation=False, llm=my_llm, tools=tools, max_rpm=10)
                
                desc = task_desc + (f"\nCONTEXTE:\n{context}" if context else "")
                task = Task(description=desc, expected_output="Court.", agent=agent)
                
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                result = crew.kickoff()
                
                # Succès
                log_container.success(f"✅ {step_name} validé par {clean_name}")
                return str(result)

            except Exception as e:
                err = str(e)
                if "429" in err or "Quota" in err:
                    match = re.search(r"retry in (\d+\.?\d*)s", err)
                    wait = float(match.group(1)) if match else 60.0
                    retry_delays.append(wait)
                    
                    log_container.warning(f"⚠️ {clean_name} épuisé (Reset: {int(wait)}s). Relais...")
                    time.sleep(1)
                    continue
                if "404" in err: continue
        
        if not retry_delays: return None
        
        if current_retry < max_retries:
            wait_time = min(retry_delays) + 2
            # Compte à rebours visuel
            for i in range(int(wait_time), 0, -1):
                log_container.info(f"⏳ Tous les agents rechargent... Reprise dans **{i}s**")
                time.sleep(1)
            current_retry += 1
        else: return None

# --- 7. ANALYSEUR TRANSPARENT ---
def analyze_one_stock(ticker, status_box, log_box):
    """Analyse une action en mettant à jour l'interface."""
    if not crew_models: return None
    
    dossier = ""
    
    # Etape 1 : Finance
    status_box.write(f"📊 **{ticker}** : L'Analyste récupère les chiffres...")
    res_fin = execute_step_smart("Finance", f"Prix, PER, Div de {ticker}.", "Analyste", [analyse_bourse_tool], crew_models, log_box)
    if not res_fin: return None
    dossier += f"FINANCE: {res_fin}\n"
    
    # Etape 2 : Sentiment
    status_box.write(f"🌍 **{ticker}** : Le Trader scanne les forums...")
    res_soc = execute_step_smart("Sentiment", f"Avis web sur {ticker}.", "Trader", [recherche_web_tool], crew_models, log_box)
    if not res_soc: res_soc = "Neutre"
    dossier += f"SENTIMENT: {res_soc}\n"
    
    # Etape 3 : Conclusion
    status_box.write(f"🧠 **{ticker}** : Le Conseiller rédige l'avis...")
    res_con = execute_step_smart("Notation", f"Analyse le dossier pour {ticker}. Note sur 10 (ex: 7/10) et avis court.", "Conseiller", [], crew_models, log_box, dossier)
    
    # Extraction note
    score = 0
    match = re.search(r"(\d+)/10", str(res_con))
    if match: score = int(match.group(1))
    
    return {"Action": ticker, "Score": score, "Note": f"{score}/10", "Avis": res_con}

def hunt_tickers(strategy_prompt, log_box):
    """Chasse avec logs."""
    if not crew_models: return []
    log_box.info("🕵️‍♂️ Le Stratège analyse les tendances de marché...")
    prompt = f"Tu es un expert en bourse. Mission : {strategy_prompt}. Donne UNIQUEMENT une liste de 4 symboles Yahoo Finance (ex: TTE.PA) séparés par des virgules."
    res = execute_step_smart("Stratège", prompt, "Stratège", [recherche_web_tool], crew_models, log_box)
    if res:
        clean = res.replace(" ", "").replace("\n", "").replace("`", "").split(",")
        return [t for t in clean if "." in t or len(t) > 2][:4]
    return []

# --- 8. INTERFACE ONGLETS ---
tab1, tab2 = st.tabs(["🔍 Analyse Solo", "🎯 Chasseur de Pépites"])

# --- TAB 1 : SOLO ---
with tab1:
    ticker_input = st.text_input("Action", "TTE.PA", key="solo_in")
    if st.button("Lancer Solo 🚀", key="solo_btn"):
        if not api_key: st.error("Clé manquante")
        else:
            # On crée un container 'status' qui s'étend
            with st.status("Démarrage de la mission...", expanded=True) as status:
                st.write("Initialisation des agents...")
                # Zone de logs techniques (Expander)
                log_box = st.expander("📟 Console des Agents (Temps réel)", expanded=True).empty()
                
                res = analyze_one_stock(ticker_input, status, log_box)
                
                if res:
                    status.update(label="Analyse Terminée !", state="complete", expanded=False)
                    st.success("Mission accomplie.")
                    st.metric(label="Note IA", value=res['Note'])
                    st.info(res['Avis'])

# --- TAB 2 : CHASSEUR ---
with tab2:
    strategy_name = st.selectbox("Stratégie :", list(HUNTING_STRATEGIES.keys()), key="hunt_sel")
    if st.button("Lancer la Chasse 🦅", key="hunt_btn"):
        if not api_key: st.error("Clé manquante")
        else:
            # Le grand container principal
            with st.status("📡 Radar Activé...", expanded=True) as main_status:
                
                # Zone de logs techniques partagée
                log_box = st.expander("📟 Journal de bord du Chasseur", expanded=True).empty()
                
                # 1. Chasse
                main_status.write("🧠 Phase 1 : Recherche de cibles...")
                tickers = hunt_tickers(HUNTING_STRATEGIES[strategy_name], log_box)
                
                if not tickers:
                    main_status.update(label="Echec recherche", state="error")
                else:
                    main_status.write(f"🎯 Cibles verrouillées : **{', '.join(tickers)}**")
                    
                    results = []
                    prog_bar = st.progress(0)
                    table_spot = st.empty() # Le tableau se construit ici
                    
                    # 2. Analyse en boucle
                    for i, t in enumerate(tickers):
                        main_status.write(f"🔬 Phase 2 ({i+1}/{len(tickers)}) : Analyse de **{t}**...")
                        
                        # Petite pause pour les yeux et le quota
                        if i > 0: 
                            log_box.info("☕ Pause tactique (3s)...")
                            time.sleep(3)
                            
                        res = analyze_one_stock(t, main_status, log_box)
                        
                        if res:
                            results.append(res)
                            # Mise à jour tableau en direct
                            df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
                            table_spot.dataframe(
                                df[["Action", "Note", "Avis"]], 
                                use_container_width=True,
                                column_config={"Avis": st.column_config.TextColumn("Verdict", width="large")}
                            )
                        
                        prog_bar.progress((i + 1) / len(tickers))

                    main_status.update(label="Chasse Terminée !", state="complete", expanded=False)
                    st.balloons()