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

# --- 1. CONFIGURATION & STYLE CSS ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

st.set_page_config(page_title="AI Financial Analyst Pro", page_icon="💎", layout="wide")

# CSS PERSONNALISÉ POUR LE RENDU "PRO"
st.markdown("""
<style>
    /* Nettoyage de l'interface par défaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Style des cartes de metrics */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    
    /* Style des boutons */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    /* Titres plus élégants */
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #E0E0E0; }
    h2 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; color: #00ADB5; }
    h3 { font-size: 1.2rem; font-weight: 500; }
    
    /* Logs Console style */
    .console-log {
        font-family: 'Courier New', monospace;
        font-size: 0.85em;
        color: #00FF41;
        background-color: #000;
        padding: 10px;
        border-radius: 5px;
        max-height: 200px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DONNÉES & STRATÉGIES ---
HUNTING_STRATEGIES = {
    "💎 Small Caps (Pépites FR)": "Trouve 4 actions françaises (PEA) small/mid caps sous-évaluées hors CAC40.",
    "🚀 Tech & Green Growth": "Trouve 4 actions européennes (PEA) Tech ou Énergie Verte en forte croissance.",
    "🛡️ Rendement Aristocrats": "Trouve 4 actions françaises solides avec dividende >5% et stable.",
    "🔥 Market Momentum": "Trouve 4 actions PEA qui buzzent positivement cette semaine."
}

# --- 3. FONCTIONS SYSTÈME (CACHE & MODELS) ---
@st.cache_data(show_spinner=False)
def get_active_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        valid = []
        for m in models:
            if 'generateContent' not in m.supported_generation_methods: continue
            if any(x in m.name.lower() for x in ['tts', 'vision', 'embedding', 'geek']): continue
            valid.append(m.name)
        # Tri : Flash 1.5 (Stable) > Flash 2.0 (Rapide) > Pro
        return sorted(valid, key=lambda x: (0 if "gemini-1.5-flash" in x else 1 if "gemini-2.0-flash" in x else 2))
    except: return []

# --- 4. SIDEBAR PRO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4205/4205906.png", width=60)
    st.markdown("### **AI Analyst Pro**")
    st.markdown("---")
    
    api_key = st.text_input("🔑 Clé API Google", type="password", help="Nécessaire pour activer les agents.")
    
    status_container = st.empty()
    if api_key:
        models = get_active_models(api_key)
        if models:
            status_container.success(f"🟢 Système Actif ({len(models)} Agents)")
            crew_models = [m.replace("models/", "gemini/") for m in models]
        else:
            status_container.error("🔴 Clé invalide ou API hors ligne")
            crew_models = []
    else:
        status_container.info("⚪ En attente de clé")
        crew_models = []
    
    st.markdown("---")
    st.markdown("#### ⚙️ Paramètres")
    with st.expander("Voir le roulement des agents"):
        if crew_models:
            for m in crew_models: st.caption(f"• {m.replace('gemini/', '')}")
        else: st.caption("Aucun agent connecté.")

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
    """Données financières."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return str({
            "Prix": info.get('currentPrice'),
            "PER": info.get('forwardPE'),
            "Div": (info.get('dividendYield', 0) or 0) * 100
        })
    except: return "Erreur Yahoo."

# --- 6. MOTEUR D'EXÉCUTION (ROBUSTE) ---
def execute_step_smart(step_name, task_desc, role, tools, model_list, log_func, context=""):
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key
    
    max_retries = 3
    current_retry = 0
    
    while current_retry <= max_retries:
        retry_delays = []
        for model_name in model_list:
            clean_name = model_name.replace("gemini/", "")
            try:
                log_func(f"⚡ **{clean_name}** travaille sur : {step_name}...", "running")
                
                my_llm = LLM(model=model_name, api_key=api_key, temperature=0.1)
                agent = Agent(role=role, goal="Expertise", backstory="Pro.", verbose=True, allow_delegation=False, llm=my_llm, tools=tools, max_rpm=10)
                
                desc = task_desc + (f"\nCONTEXTE PRÉCÉDENT:\n{context}" if context else "")
                task = Task(description=desc, expected_output="Synthétique.", agent=agent)
                
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                result = crew.kickoff()
                
                log_func(f"✅ {step_name} validé.", "success")
                return str(result)

            except Exception as e:
                err = str(e)
                if "429" in err or "Quota" in err:
                    match = re.search(r"retry in (\d+\.?\d*)s", err)
                    wait = float(match.group(1)) if match else 60.0
                    retry_delays.append(wait)
                    log_func(f"⚠️ {clean_name} Quota (Reset: {int(wait)}s). Bascule...", "warning")
                    time.sleep(1)
                    continue
                if "404" in err: continue
        
        if not retry_delays: return None
        if current_retry < max_retries:
            wait_time = min(retry_delays) + 2
            for i in range(int(wait_time), 0, -1):
                log_func(f"🛑 Pause Quota Global... Reprise {i}s", "error")
                time.sleep(1)
            current_retry += 1
        else: return None

# --- 7. LOGIQUE MÉTIER ---
def parse_finance_data(fin_str):
    """Extrait proprement les chiffres du string brut."""
    try:
        # Nettoyage basique si l'IA bavarde
        fin_str = fin_str.replace("'", '"')
        match_prix = re.search(r"'?Prix'?: ?([\d\.]+)", fin_str)
        match_per = re.search(r"'?PER'?: ?([\d\.]+)", fin_str)
        match_div = re.search(r"'?Div'?: ?([\d\.]+)", fin_str)
        
        return {
            "Prix": float(match_prix.group(1)) if match_prix else 0.0,
            "PER": float(match_per.group(1)) if match_per else 0.0,
            "Div": float(match_div.group(1)) if match_div else 0.0
        }
    except:
        return {"Prix": 0, "PER": 0, "Div": 0}

def analyze_one_stock(ticker, update_status_func):
    """Analyse complète."""
    if not crew_models: return None
    
    dossier = ""
    # Logs interne
    log_history = []
    def log_wrapper(msg, state):
        log_history.append(f"[{state.upper()}] {msg}")
        update_status_func(msg, state)

    # 1. Finance
    res_fin = execute_step_smart("Finance", f"Donne Prix, PER, Div (%) de {ticker}.", "Analyste", [analyse_bourse_tool], crew_models, log_wrapper)
    if not res_fin: return None
    fin_data = parse_finance_data(res_fin)
    dossier += f"FINANCE: {res_fin}\n"
    
    # 2. Sentiment
    res_soc = execute_step_smart("Sentiment", f"Avis web sur {ticker}.", "Trader", [recherche_web_tool], crew_models, log_wrapper)
    if not res_soc: res_soc = "Neutre"
    dossier += f"SENTIMENT: {res_soc}\n"
    
    # 3. Verdict
    res_con = execute_step_smart("Notation", f"Analyse le dossier {ticker}. Donne note /10 et avis court (ACHAT/VENTE/ATTENTE).", "Conseiller", [], crew_models, log_wrapper, dossier)
    
    score = 0
    match = re.search(r"(\d+)/10", str(res_con))
    if match: score = int(match.group(1))
    
    # Détection recommandation
    rec = "NEUTRE"
    if "ACHAT" in res_con.upper(): rec = "ACHAT"
    elif "VENTE" in res_con.upper(): rec = "VENTE"
    
    return {
        "Ticker": ticker,
        "Data": fin_data,
        "Score": score,
        "Rec": rec,
        "Avis": res_con,
        "Logs": log_history
    }

def hunt_tickers(strategy_prompt, status_func):
    status_func("🧠 Le Stratège scanne le marché...", "running")
    prompt = f"Expert bourse. Mission: {strategy_prompt}. Donne UNIQUEMENT liste de 4 tickers Yahoo (ex: AI.PA) séparés par virgules."
    res = execute_step_smart("Stratège", prompt, "Stratège", [recherche_web_tool], crew_models, lambda m,s: None) # Pas de log détail ici
    if res:
        clean = res.replace(" ", "").replace("\n", "").split(",")
        return [t for t in clean if "." in t or len(t) > 2][:4]
    return []

# --- 8. INTERFACE PRINCIPALE (UI DASHBOARD) ---

# Header personnalisé
st.markdown("## 💎 AI Financial Analyst <span style='font-size:0.6em; color:gray'>v3.0 Pro</span>", unsafe_allow_html=True)

# Tabs stylisés
tab_solo, tab_radar = st.tabs(["🔍 Analyse Focus", "📡 Radar de Marché"])

# === TAB 1 : FOCUS (SOLO) ===
with tab_solo:
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        ticker_input = st.text_input("Symbole de l'action", value="TTE.PA", placeholder="Ex: MC.PA, NVDA", label_visibility="collapsed")
    with col_btn:
        btn_solo = st.button("Lancer l'Analyse", key="btn_s")

    if btn_solo and api_key:
        # Zone d'affichage des résultats
        result_container = st.container()
        
        # Zone de status (Logs UX)
        with st.status("🚀 Analyse en cours...", expanded=True) as status:
            log_placeholder = st.empty()
            
            def update_ui_log(msg, type_msg):
                if type_msg == "running": log_placeholder.info(msg)
                elif type_msg == "success": log_placeholder.success(msg)
                elif type_msg == "warning": log_placeholder.warning(msg)
                elif type_msg == "error": log_placeholder.error(msg)
            
            res = analyze_one_stock(ticker_input, update_ui_log)
            
            if res:
                status.update(label="✅ Analyse terminée avec succès", state="complete", expanded=False)
                
                # --- AFFICHAGE PRO DES RÉSULTATS ---
                with result_container:
                    st.markdown("---")
                    # En-tête avec le Score géant
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1:
                        st.metric("Score IA", f"{res['Score']}/10", delta="Potentiel", delta_color="normal")
                    with c2:
                        rec_color = "green" if res['Rec'] == "ACHAT" else "red" if res['Rec'] == "VENTE" else "gray"
                        st.markdown(f"<h2 style='text-align: center; color: {rec_color}; margin-top: 0;'>{res['Rec']}</h2>", unsafe_allow_html=True)
                        st.caption("Recommandation basée sur Finance + Sentiment")
                    with c3:
                         st.metric("Confiance", "Haute")

                    # KPIs Financiers (Cartes)
                    st.markdown("##### 📊 Indicateurs Clés")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Prix Actuel", f"{res['Data']['Prix']} €")
                    k2.metric("P.E.R (Ratio)", f"{res['Data']['PER']:.1f}")
                    k3.metric("Rendement Div.", f"{res['Data']['Div']:.2f} %")
                    
                    # Analyse textuelle
                    st.info(f"💡 **Avis de l'Expert :** {res['Avis']}")
                    
                    # Logs techniques (Cachés mais dispos)
                    with st.expander("🛠️ Voir le journal technique"):
                        st.code("\n".join(res['Logs']), language="bash")

# === TAB 2 : RADAR (CHASSEUR) ===
with tab_radar:
    c_strat, c_go = st.columns([3, 1])
    with c_strat:
        strat_choice = st.selectbox("Stratégie de Chasse", list(HUNTING_STRATEGIES.keys()), label_visibility="collapsed")
    with c_go:
        btn_radar = st.button("Activer le Radar", key="btn_r")

    if btn_radar and api_key:
        st.markdown("---")
        radar_status = st.empty()
        radar_prog = st.progress(0)
        
        # Zone du tableau vide au début
        table_spot = st.empty()
        
        # Fonction log locale
        def radar_log(msg, type_msg):
            radar_status.caption(f"📡 {msg}")

        # 1. Chasse
        tickers = hunt_tickers(HUNTING_STRATEGIES[strat_choice], radar_log)
        
        if not tickers:
            st.error("Aucune cible trouvée.")
        else:
            radar_status.markdown(f"**🎯 Cibles :** `{'` `'.join(tickers)}`")
            results = []
            
            for i, t in enumerate(tickers):
                # Pause tactique UI
                if i > 0: 
                    radar_log(f"Pause tactique avant {t}...", "running")
                    time.sleep(3)
                    
                # Analyse silencieuse (on ne montre pas le détail ici)
                res = analyze_one_stock(t, radar_log)
                
                if res:
                    results.append({
                        "Action": res['Ticker'],
                        "Score": res['Score'],
                        "Avis": res['Rec'], # Juste le mot clé
                        "Prix": f"{res['Data']['Prix']} €",
                        "PER": f"{res['Data']['PER']:.1f}",
                        "Rendement": f"{res['Data']['Div']:.1f}%",
                        "Détail": res['Avis'] # Pour le tooltip ou expander
                    })
                    
                    # MISE A JOUR DU TABLEAU EN TEMPS REEL
                    df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
                    
                    table_spot.dataframe(
                        df,
                        column_order=("Action", "Score", "Avis", "Prix", "PER", "Rendement", "Détail"),
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Score": st.column_config.ProgressColumn(
                                "Potentiel",
                                help="Score sur 10",
                                format="%d/10",
                                min_value=0,
                                max_value=10,
                            ),
                            "Avis": st.column_config.TextColumn(
                                "Verdict",
                                help="Recommandation IA",
                                width="small"
                            ),
                            "Détail": st.column_config.TextColumn(
                                "Analyse Complète",
                                width="large"
                            )
                        }
                    )
                
                radar_prog.progress((i + 1) / len(tickers))
            
            radar_status.success("Radar terminé !")
            st.balloons()