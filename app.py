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

st.set_page_config(page_title="AI Financial Analyst", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stMetric"] { background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px; }
    .stProgress > div > div > div > div { background-color: #00ADB5; }
</style>
""", unsafe_allow_html=True)

# --- 2. LISTES DE SECOURS (Plan B immédiat) ---
FALLBACK_LISTS = {
    "💎 Small Caps (Pépites FR)": ["VLA.PA", "ATO.PA", "SESL.PA", "BEN.PA"],
    "🚀 Tech & Green Growth": ["SOIT.PA", "AI.PA", "DSY.PA", "STM.PA"],
    "🛡️ Rendement Aristocrats": ["ACA.PA", "BNP.PA", "ENGI.PA", "RNO.PA"],
    "🔥 Market Momentum": ["MC.PA", "RMS.PA", "OR.PA", "AIR.PA"]
}

HUNTING_STRATEGIES = {
    "💎 Small Caps (Pépites FR)": "Trouve 4 actions françaises small/mid caps PEA hors CAC40.",
    "🚀 Tech & Green Growth": "Trouve 4 actions européennes PEA Tech/Green en croissance.",
    "🛡️ Rendement Aristocrats": "Trouve 4 actions françaises PEA avec gros dividende stable.",
    "🔥 Market Momentum": "Trouve 4 actions PEA populaires cette semaine."
}

# --- 3. FILTRE MODÈLES ---
@st.cache_data(show_spinner=False)
def get_active_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        valid = []
        BANNED = ['tts', 'vision', 'embedding', 'geek', 'gecko', 'image', 'banana', 'nano']
        for m in models:
            if 'generateContent' not in m.supported_generation_methods: continue
            if any(b in m.name.lower() for b in BANNED): continue
            valid.append(m.name)
        return sorted(valid, key=lambda x: (0 if "gemini-1.5-flash" in x and "8b" not in x else 1 if "gemini-2.0-flash" in x else 2))
    except: return []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚡ AI Analyst Commando")
    api_key = st.text_input("Clé API Google", type="password")
    crew_models = []
    if api_key:
        models = get_active_models(api_key)
        if models:
            st.success(f"🟢 {len(models)} Modèles Prêts")
            crew_models = [m.replace("models/", "gemini/") for m in models]
        else: st.error("Aucun modèle valide.")

# --- 5. OUTILS OPTIMISÉS ---
@tool("Recherche Web")
def recherche_web_tool(query: str):
    """Recherche Web précise."""
    try:
        with DDGS() as ddgs:
            # On demande plus de résultats mais on filtre les déchets
            results = list(ddgs.text(query, max_results=4))
            clean_results = []
            for r in results:
                # Filtre anti-dictionnaire/pub
                if "dictionary" in r['title'].lower() or "auction" in r['body'].lower(): continue
                clean_results.append(f"- {r['body']}")
            
            return "\n".join(clean_results[:2]) if clean_results else "Pas d'info pertinente."
    except: return "Erreur recherche."

@tool("Bourse Yahoo")
def analyse_bourse_tool(ticker: str):
    """Données financières."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        p = info.get('currentPrice') or info.get('regularMarketPrice') or 0.0
        r = info.get('forwardPE') or info.get('trailingPE') or 0.0
        d = (info.get('dividendYield', 0) or 0) * 100
        return str({"Prix": round(p, 2), "PER": round(r, 2), "Div": round(d, 2)})
    except: return str({"Prix": 0, "PER": 0, "Div": 0})

# --- 6. MOTEUR RAPIDE (MAX ITERATIONS) ---
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
                log_func(f"⚡ {clean_name}...", "running")
                my_llm = LLM(model=model_name, api_key=api_key, temperature=0.1)
                
                # --- OPTIMISATION MAJEURE : MAX_ITER=2 ---
                # L'agent n'a le droit qu'à 2 essais. S'il échoue, il arrête de boucler.
                agent = Agent(
                    role=role, 
                    goal="Réponse immédiate.", 
                    backstory="Tu vas droit au but. Pas de bla-bla.", 
                    verbose=True, 
                    allow_delegation=False, 
                    llm=my_llm, 
                    tools=tools, 
                    max_rpm=100,
                    max_iter=2 # <--- ICI : On empêche les boucles infinies
                )
                
                desc = task_desc + (f"\nCONTEXTE:\n{context}" if context else "")
                task = Task(description=desc, expected_output="Très court.", agent=agent)
                
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                result = crew.kickoff()
                log_func(f"✅ {step_name} OK.", "success")
                return str(result)

            except Exception as e:
                err = str(e)
                if "429" in err or "Quota" in err:
                    match = re.search(r"retry in (\d+\.?\d*)s", err)
                    wait = float(match.group(1)) if match else 20.0
                    retry_delays.append(wait)
                    log_func(f"⚠️ {clean_name} vide.", "warning")
                    continue
                if "404" in err or "400" in err: continue
        
        if not retry_delays: return None
        
        if current_retry < max_retries:
            wait_time = min(retry_delays) + 2
            # Animation rapide
            for i in range(int(wait_time), 0, -1):
                log_func(f"⏳ Attente Quota ({i}s)...", "error")
                time.sleep(1)
            current_retry += 1
        else: return None

# --- 7. LOGIQUE MÉTIER ---
def hunt_tickers(strategy_name, strategy_prompt, status_func):
    status_func(f"🧠 Recherche rapide : {strategy_name}...", "running")
    
    # PROMPT ENFANTIN ET STRICT
    prompt = f"""
    Trouve 4 symboles boursiers (Tickers Yahoo) pour : {strategy_prompt}
    RÈGLES STRICTES :
    1. Réponds JUSTE la liste séparée par des virgules.
    2. Exemple : TTE.PA, AIR.PA, BNP.PA
    3. PAS DE PHRASES. PAS D'EXPLICATION.
    4. Si tu ne trouves pas, invente rien.
    """
    
    res = execute_step_smart("Stratège", prompt, "Stratège", [recherche_web_tool], crew_models, lambda m,s: None)
    
    tickers = []
    if res:
        # Extraction brutale
        found = re.findall(r'\b[A-Z0-9]{2,5}\.[A-Z]{2,3}\b', res.upper())
        tickers = list(set(found))[:4]
    
    # PLAN B IMMÉDIAT (Si l'IA échoue ou prend trop de temps, on injecte la liste manuelle)
    if not tickers:
        status_func("⚠️ IA confuse, activation liste de secours ultra-rapide...", "warning")
        time.sleep(1)
        tickers = FALLBACK_LISTS.get(strategy_name, ["TTE.PA", "AIR.PA", "BNP.PA", "SAN.PA"])
    
    return tickers

def parse_finance_data(fin_str):
    try:
        clean_str = fin_str.replace("'", "").replace('"', '').replace(":", " ")
        p = float(re.search(r"(?:Prix|Price)[\s]+([\d\.]+)", clean_str, re.IGNORECASE).group(1)) if re.search(r"(?:Prix|Price)", clean_str, re.IGNORECASE) else 0.0
        r = float(re.search(r"(?:PER|P/E)[\s]+([\d\.]+)", clean_str, re.IGNORECASE).group(1)) if re.search(r"(?:PER|P/E)", clean_str, re.IGNORECASE) else 0.0
        d = float(re.search(r"(?:Div|Yield)[\s]+([\d\.]+)", clean_str, re.IGNORECASE).group(1)) if re.search(r"(?:Div|Yield)", clean_str, re.IGNORECASE) else 0.0
        return {"Prix": p, "PER": r, "Div": d}
    except: return {"Prix": 0.0, "PER": 0.0, "Div": 0.0}

def analyze_one_stock(ticker, update_status_func):
    if not crew_models: return None
    dossier = ""
    def log_wrapper(msg, state): update_status_func(msg, state)

    # 1. Finance (Prompt JSON strict)
    prompt_fin = f"Pour {ticker}, donne moi le JSON : {{'Prix': 0.0, 'PER': 0.0, 'Div': 0.0}}. Trouve les vrais chiffres."
    res_fin = execute_step_smart("Finance", prompt_fin, "Analyste", [analyse_bourse_tool], crew_models, log_wrapper)
    if not res_fin: return None
    fin_data = parse_finance_data(res_fin)
    dossier += f"FINANCE: {res_fin}\n"
    
    # 2. Sentiment
    res_soc = execute_step_smart("Sentiment", f"Donne moi l'avis web sur {ticker} en 1 phrase.", "Trader", [recherche_web_tool], crew_models, log_wrapper)
    if not res_soc: res_soc = "Pas d'info"
    dossier += f"SENTIMENT: {res_soc}\n"
    
    # 3. Verdict
    res_con = execute_step_smart("Notation", f"Lis le dossier {ticker}. Note /10 et dis ACHAT ou VENTE ou ATTENTE. 1 phrase max.", "Conseiller", [], crew_models, log_wrapper, dossier)
    
    score = 0
    match = re.search(r"(\d+)/10", str(res_con))
    if match: score = int(match.group(1))
    
    rec = "NEUTRE"
    if "ACHAT" in res_con.upper(): rec = "ACHAT"
    elif "VENTE" in res_con.upper(): rec = "VENTE"
    
    return {"Ticker": ticker, "Data": fin_data, "Score": score, "Rec": rec, "Avis": res_con}

# --- 8. INTERFACE ---
st.markdown("## ⚡ AI Financial Analyst <span style='font-size:0.6em; color:gray'>Turbo</span>", unsafe_allow_html=True)
tab_solo, tab_radar = st.tabs(["🔍 Analyse", "📡 Radar"])

with tab_solo:
    c1, c2 = st.columns([3, 1])
    ticker_in = c1.text_input("Symbole", "TTE.PA", label_visibility="collapsed")
    if c2.button("GO", key="btn_s") and api_key:
        with st.status("Traitement...", expanded=True) as status:
            log_spot = st.empty()
            res = analyze_one_stock(ticker_in, lambda m,t: log_spot.info(m) if t=="running" else log_spot.success(m) if t=="success" else log_spot.warning(m))
            if res:
                status.update(label="Terminé", state="complete", expanded=False)
                st.markdown("---")
                k1, k2, k3 = st.columns([1,2,1])
                k1.metric("Score", f"{res['Score']}/10")
                k2.markdown(f"<h2 style='text-align:center;color:{'#00FF41' if 'ACHAT' in res['Rec'] else '#FF4136'}'>{res['Rec']}</h2>", unsafe_allow_html=True)
                k3.metric("Confiance", "Haute")
                d1, d2, d3 = st.columns(3)
                d1.metric("Prix", f"{res['Data']['Prix']} €")
                d2.metric("PER", f"{res['Data']['PER']:.1f}")
                d3.metric("Div", f"{res['Data']['Div']:.2f} %")
                st.info(f"**Avis:** {res['Avis']}")

with tab_radar:
    c_s, c_b = st.columns([3, 1])
    strat_key = c_s.selectbox("Stratégie", list(HUNTING_STRATEGIES.keys()), label_visibility="collapsed")
    
    if c_b.button("SCAN", key="btn_r") and api_key:
        st.markdown("---")
        radar_stat = st.empty()
        radar_bar = st.progress(0)
        table_spot = st.empty()
        
        def r_log(m, t): radar_stat.caption(f"📡 {m}")
        
        tickers = hunt_tickers(strat_key, HUNTING_STRATEGIES[strat_key], r_log)
        
        radar_stat.markdown(f"**Cibles:** `{'` `'.join(tickers)}`")
        data = []
        
        for i, t in enumerate(tickers):
            if i > 0: time.sleep(2) # Petite pause tactique
            res = analyze_one_stock(t, r_log)
            if res:
                data.append({
                    "Action": t, 
                    "Score": res['Score'], 
                    "Avis": res['Rec'],
                    "Prix": f"{res['Data']['Prix']} €", 
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