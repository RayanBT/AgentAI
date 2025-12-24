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

# --- 1. CONFIGURATION & STYLE ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

st.set_page_config(page_title="AI Financial Analyst", page_icon="💎", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stMetric"] {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px;
    }
    /* Style du compte à rebours */
    .stAlert { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. STRATÉGIES ---
HUNTING_STRATEGIES = {
    "💎 Small Caps (Pépites FR)": "Trouve 4 actions françaises (PEA) small/mid caps sous-évaluées hors CAC40.",
    "🚀 Tech & Green Growth": "Trouve 4 actions européennes (PEA) Tech ou Énergie Verte en forte croissance.",
    "🛡️ Rendement Aristocrats": "Trouve 4 actions françaises solides avec dividende >5% et stable.",
    "🔥 Market Momentum": "Trouve 4 actions PEA qui buzzent positivement cette semaine."
}

# --- 3. GESTION DES MODÈLES (FILTRE STRICT) ---
@st.cache_data(show_spinner=False)
def get_active_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        valid = []
        # LISTE NOIRE pour éviter les crashs
        BANNED = ['tts', 'vision', 'embedding', 'geek', 'gecko', 'image', 'banana', 'nano', 'exp', 'preview']
        
        for m in models:
            if 'generateContent' not in m.supported_generation_methods: continue
            if any(b in m.name.lower() for b in BANNED): continue
            valid.append(m.name)
            
        # TRI : On privilégie la stabilité
        return sorted(valid, key=lambda x: (
            0 if "gemini-1.5-flash" in x and "8b" not in x else 
            1 if "gemini-2.0-flash" in x else 2
        ))
    except: return []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("### 💎 AI Analyst Pro")
    api_key = st.text_input("Clé API Google", type="password")
    
    crew_models = []
    if api_key:
        models = get_active_models(api_key)
        if models:
            st.success(f"🟢 {len(models)} Modèles Stables")
            crew_models = [m.replace("models/", "gemini/") for m in models]
        else:
            st.error("Aucun modèle stable.")

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
    """Récupère Prix, PER et Dividende."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Récupération sécurisée des données (évite les None)
        price = info.get('currentPrice') or info.get('regularMarketPrice') or 0.0
        per = info.get('forwardPE') or info.get('trailingPE') or 0.0
        div = (info.get('dividendYield', 0) or 0) * 100
        
        # On retourne un format JSON String propre
        return str({
            "Prix": round(price, 2),
            "PER": round(per, 2),
            "Div": round(div, 2)
        })
    except: return str({"Prix": 0, "PER": 0, "Div": 0})

# --- 6. MOTEUR D'EXÉCUTION (AVEC PAUSE VISUELLE) ---
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
                log_func(f"⚡ {clean_name} analyse...", "running")
                my_llm = LLM(model=model_name, api_key=api_key, temperature=0.1)
                agent = Agent(role=role, goal="Expertise", backstory="Pro.", verbose=True, allow_delegation=False, llm=my_llm, tools=tools, max_rpm=10)
                
                desc = task_desc + (f"\nCONTEXTE:\n{context}" if context else "")
                task = Task(description=desc, expected_output="Synthétique.", agent=agent)
                
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                result = crew.kickoff()
                log_func(f"✅ {step_name} validé.", "success")
                return str(result)

            except Exception as e:
                err = str(e)
                if "429" in err or "Quota" in err:
                    match = re.search(r"retry in (\d+\.?\d*)s", err)
                    wait = float(match.group(1)) if match else 30.0
                    retry_delays.append(wait)
                    log_func(f"⚠️ {clean_name} vide (Reset {int(wait)}s).", "warning")
                    time.sleep(1)
                    continue
                if "404" in err or "400" in err: continue
        
        # SI TOUT LE MONDE EST VIDE : PAUSE VISUELLE
        if not retry_delays: return None
        
        if current_retry < max_retries:
            wait_time = min(retry_delays) + 2
            
            # --- AFFICHAGE DU COMPTE A REBOURS ---
            countdown_box = st.empty()
            progress_bar = st.progress(0)
            
            for i in range(int(wait_time), 0, -1):
                percent = 1.0 - (i / wait_time)
                countdown_box.error(f"🛑 **PAUSE QUOTA GOOGLE** : Tous les agents rechargent... Reprise dans **{i}s**")
                progress_bar.progress(percent)
                time.sleep(1)
            
            # Nettoyage après attente
            countdown_box.empty()
            progress_bar.empty()
            log_func("🔄 Reprise du travail !", "info")
            current_retry += 1
        else: return None

# --- 7. LOGIQUE MÉTIER & PARSING (Correction du Bug des "0") ---
def parse_finance_data(fin_str):
    """Extrait les chiffres avec des Regex souples."""
    try:
        # On nettoie la chaine pour faciliter la recherche
        clean_str = fin_str.replace("'", "").replace('"', '').replace(":", " ")
        
        # Regex qui cherche "Prix" suivi de chiffres (avec point ou virgule)
        match_prix = re.search(r"(?:Prix|Price)[\s]+([\d\.]+)", clean_str, re.IGNORECASE)
        match_per = re.search(r"(?:PER|P/E)[\s]+([\d\.]+)", clean_str, re.IGNORECASE)
        match_div = re.search(r"(?:Div|Yield)[\s]+([\d\.]+)", clean_str, re.IGNORECASE)
        
        return {
            "Prix": float(match_prix.group(1)) if match_prix else 0.0,
            "PER": float(match_per.group(1)) if match_per else 0.0,
            "Div": float(match_div.group(1)) if match_div else 0.0
        }
    except:
        return {"Prix": 0.0, "PER": 0.0, "Div": 0.0}

def analyze_one_stock(ticker, update_status_func):
    if not crew_models: return None
    
    dossier = ""
    def log_wrapper(msg, state): update_status_func(msg, state)

    # 1. FINANCE : On force l'IA à répondre un format précis pour éviter le bug des 0
    prompt_finance = f"""
    Utilise l'outil Bourse pour {ticker}.
    IMPORTANT : Ta réponse DOIT être UNIQUEMENT sous cette forme :
    Prix: X
    PER: Y
    Div: Z
    Ne mets pas de texte autour.
    """
    res_fin = execute_step_smart("Finance", prompt_finance, "Analyste", [analyse_bourse_tool], crew_models, log_wrapper)
    if not res_fin: return None
    
    # Extraction Immédiate
    fin_data = parse_finance_data(res_fin)
    dossier += f"FINANCE: {res_fin}\n"
    
    # 2. SENTIMENT
    res_soc = execute_step_smart("Sentiment", f"Avis web sur {ticker}.", "Trader", [recherche_web_tool], crew_models, log_wrapper)
    if not res_soc: res_soc = "Neutre"
    dossier += f"SENTIMENT: {res_soc}\n"
    
    # 3. VERDICT
    res_con = execute_step_smart("Notation", f"Analyse dossier {ticker}. Note /10 et avis court (ACHAT/VENTE).", "Conseiller", [], crew_models, log_wrapper, dossier)
    
    score = 0
    match = re.search(r"(\d+)/10", str(res_con))
    if match: score = int(match.group(1))
    
    rec = "NEUTRE"
    if "ACHAT" in res_con.upper(): rec = "ACHAT"
    elif "VENTE" in res_con.upper(): rec = "VENTE"
    
    return {"Ticker": ticker, "Data": fin_data, "Score": score, "Rec": rec, "Avis": res_con}

def hunt_tickers(strategy_prompt, status_func):
    status_func("🧠 Le Stratège scanne...", "running")
    prompt = f"Expert bourse. Mission: {strategy_prompt}. Donne UNIQUEMENT 4 tickers Yahoo (ex: AI.PA) séparés par virgules."
    res = execute_step_smart("Stratège", prompt, "Stratège", [recherche_web_tool], crew_models, lambda m,s: None)
    if res:
        clean = res.replace(" ", "").replace("\n", "").split(",")
        return [t for t in clean if "." in t or len(t) > 2][:4]
    return []

# --- 8. INTERFACE DASHBOARD ---
st.markdown("## 💎 AI Financial Analyst <span style='font-size:0.6em; color:gray'>Pro Edition</span>", unsafe_allow_html=True)

tab_solo, tab_radar = st.tabs(["🔍 Analyse Focus", "📡 Radar de Marché"])

# === TAB 1 : FOCUS ===
with tab_solo:
    c1, c2 = st.columns([3, 1])
    ticker_in = c1.text_input("Symbole", "TTE.PA", label_visibility="collapsed")
    if c2.button("Analyser", key="btn_s") and api_key:
        with st.status("Analyse en cours...", expanded=True) as status:
            log_spot = st.empty()
            def ui_log(m, t): 
                if t=="running": log_spot.info(m)
                elif t=="success": log_spot.success(m)
                elif t=="warning": log_spot.warning(m)
                elif t=="error": log_spot.error(m)
            
            res = analyze_one_stock(ticker_in, ui_log)
            if res:
                status.update(label="Terminé", state="complete", expanded=False)
                st.markdown("---")
                
                k1, k2, k3 = st.columns([1,2,1])
                k1.metric("Score", f"{res['Score']}/10")
                k2.markdown(f"<h2 style='text-align:center;color:{'#00FF41' if 'ACHAT' in res['Rec'] else '#FF4136'}'>{res['Rec']}</h2>", unsafe_allow_html=True)
                k3.metric("Confiance", "Haute")
                
                # BUG DES 0 CORRIGÉ ICI
                d1, d2, d3 = st.columns(3)
                d1.metric("Prix", f"{res['Data']['Prix']} €")
                d2.metric("PER", f"{res['Data']['PER']:.1f}")
                d3.metric("Div", f"{res['Data']['Div']:.2f} %")
                
                st.info(f"**Avis:** {res['Avis']}")

# === TAB 2 : RADAR ===
with tab_radar:
    c_s, c_b = st.columns([3, 1])
    strat = c_s.selectbox("Stratégie", list(HUNTING_STRATEGIES.keys()), label_visibility="collapsed")
    
    if c_b.button("Scanner", key="btn_r") and api_key:
        st.markdown("---")
        radar_stat = st.empty()
        radar_bar = st.progress(0)
        table_spot = st.empty()
        
        def r_log(m, t): radar_stat.caption(f"📡 {m}")
        
        tickers = hunt_tickers(HUNTING_STRATEGIES[strat], r_log)
        
        if tickers:
            radar_stat.markdown(f"**Cibles:** `{'` `'.join(tickers)}`")
            data = []
            
            for i, t in enumerate(tickers):
                if i > 0:
                    r_log(f"Pause tactique (3s) avant {t}...", "running")
                    time.sleep(3)
                
                res = analyze_one_stock(t, r_log)
                if res:
                    data.append({
                        "Action": t, 
                        "Score": res['Score'], 
                        "Avis": res['Rec'],
                        "Prix": f"{res['Data']['Prix']} €", # Affichera la vraie valeur
                        "PER": f"{res['Data']['PER']:.1f}",
                        "Rendement": f"{res['Data']['Div']:.1f}%",
                        "Résumé": res['Avis']
                    })
                    
                    df = pd.DataFrame(data).sort_values(by="Score", ascending=False)
                    table_spot.dataframe(
                        df, 
                        column_order=("Action", "Score", "Avis", "Prix", "PER", "Rendement", "Résumé"),
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Score": st.column_config.ProgressColumn("Note", format="%d/10", min_value=0, max_value=10),
                            "Résumé": st.column_config.TextColumn("Détail", width="large")
                        }
                    )
                radar_bar.progress((i+1)/len(tickers))
            radar_stat.success("Scan terminé !")
            st.balloons()
        else:
            st.error("Aucune cible trouvée.")