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

# --- 1. CONFIGURATION SYSTEME ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

st.set_page_config(page_title="Agent PEA Stratège", page_icon="🏯", layout="wide")

st.title("🏯 Assistant PEA (Roulement Stratégique)")
st.markdown("Système de **chasse aux actions** avec gestion automatique des quotas Google (Basculement de modèle).")

# --- 2. STRATÉGIES DE CHASSE ---
HUNTING_STRATEGIES = {
    "💎 Pépites Cachées (Small Caps)": "Trouve 4 actions françaises (PEA) de petite/moyenne capitalisation sous-évaluées. Cherche hors du CAC40.",
    "🚀 Croissance Tech/Verte": "Trouve 4 actions européennes (PEA) Tech ou Green Energy avec forte croissance.",
    "🛡️ Rendement & Dividende": "Trouve 4 actions françaises solides avec un haut rendement de dividende (>5%) et stable.",
    "🔥 Momentum (Buzz actuel)": "Trouve 4 actions PEA qui font l'actualité positivement cette semaine."
}

# --- 3. FONCTIONS UTILITAIRES (CACHE & FILTRAGE) ---
@st.cache_data(show_spinner=False)
def get_active_models(api_key):
    """
    Récupère les modèles et les trie pour utiliser les plus généreux en premier.
    """
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        valid_models = []
        
        for m in models:
            name = m.name.lower()
            # Filtre strict anti-crash (pas d'audio, pas de vision, pas d'embedding)
            if 'generateContent' not in m.supported_generation_methods: continue
            if any(x in name for x in ['tts', 'vision', 'embedding', 'geek', 'gecko']): continue
            valid_models.append(m.name)
            
        # TRI STRATÉGIQUE : On privilégie la stabilité et les quotas larges
        # 1. Flash 1.5 (Souvent le plus large en quota)
        # 2. Flash 2.0 (Rapide mais parfois limité)
        # 3. Pro (Plus lent)
        return sorted(valid_models, key=lambda x: (
            0 if "gemini-1.5-flash" in x and "8b" not in x else  # Le plus fiable en quota
            1 if "gemini-2.0-flash" in x else                    # Le plus rapide
            2 if "flash" in x else                               # Les autres Flash
            3                                                    # Les Pro
        ))
    except: return []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé Google API", type="password")
    
    crew_models = []
    if api_key:
        models = get_active_models(api_key)
        if models:
            crew_models = [m.replace("models/", "gemini/") for m in models]
            st.success(f"✅ {len(models)} modèles chargés en relais")
            with st.expander("Voir l'ordre de roulement"):
                for i, m in enumerate(crew_models):
                    st.caption(f"{i+1}. {m}")
        else:
            st.error("Aucun modèle valide.")

# --- 5. OUTILS ---
@tool("Recherche Web")
def recherche_web_tool(query: str):
    """Recherche Web."""
    try:
        with DDGS() as ddgs:
            # Limité à 2 résultats pour économiser les tokens
            results = list(ddgs.text(query, max_results=2))
            return "\n".join([f"- {r['body']}" for r in results]) if results else "Rien trouvé."
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

# --- 6. MOTEUR ROBUSTE (AVEC ROULEMENT) ---
def execute_step_smart(step_name, task_desc, role, tools, model_list, context=""):
    """
    Exécute une tâche. Si un modèle échoue, passe au suivant.
    """
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key
    
    # On essaie les modèles en cascade
    for model_name in model_list:
        clean_name = model_name.replace("gemini/", "")
        
        try:
            # 1. Config Agent
            my_llm = LLM(model=model_name, api_key=api_key, temperature=0.1)
            agent = Agent(role=role, goal="Tâche", backstory="Expert.", verbose=True, allow_delegation=False, llm=my_llm, tools=tools, max_rpm=10)
            
            # 2. Config Tâche
            desc = task_desc + (f"\nCONTEXTE:\n{context}" if context else "")
            task = Task(description=desc, expected_output="Court et précis.", agent=agent)
            
            # 3. Exécution
            crew = Crew(agents=[agent], tasks=[task], verbose=True)
            result = crew.kickoff()
            
            # Si succès, on affiche un petit toast discret et on renvoie
            st.toast(f"✅ {step_name} OK ({clean_name})", icon="🔹")
            return str(result)

        except Exception as e:
            err = str(e)
            
            # GESTION DES ERREURS
            if "404" in err or "400" in err: 
                continue # Modèle invalide, on zappe
            
            if "429" in err or "Quota" in err or "ResourceExhausted" in err:
                # Quota atteint ! On prévient et on passe au suivant
                st.toast(f"⚠️ {clean_name} vide. Relais vers le suivant...", icon="⏩")
                time.sleep(1) # Petite pause technique
                continue
            
            # Autres erreurs
            print(f"Erreur sur {clean_name}: {e}")
            continue

    # Si on arrive ici, TOUS les modèles ont échoué
    st.error(f"❌ Échec total de l'étape '{step_name}'. Tous les quotas sont pleins.")
    return None

# --- 7. FONCTIONS MÉTIER ---
def hunt_tickers(strategy_prompt):
    """Trouve les actions."""
    if not crew_models: return []
    
    prompt = f"Tu es un expert en bourse. Mission : {strategy_prompt}. Donne UNIQUEMENT une liste de 4 symboles Yahoo Finance valides (ex: TTE.PA, AIR.PA) séparés par des virgules. Pas de texte explicatif."
    
    # On utilise le moteur pour trouver les tickers
    res = execute_step_smart("Stratège", prompt, "Stratège", [recherche_web_tool], crew_models)
    
    if res:
        # Nettoyage agressif pour avoir une liste propre
        clean = res.replace(" ", "").replace("\n", "").replace("`", "").replace("'", "").split(",")
        # Filtre pour garder les formats boursiers
        final_list = [t.strip() for t in clean if "." in t or len(t.strip()) > 2]
        return final_list[:4] # On limite à 4 actions max pour survivre aux quotas
    return []

def analyze_one_stock(ticker, progress_callback=None):
    """Analyse une action en 3 étapes sécurisées."""
    if not crew_models: return None
    
    dossier = ""
    
    # Etape 1 : Finance
    if progress_callback: progress_callback(f"🔎 {ticker} : Données financières...")
    res_fin = execute_step_smart("Finance", f"Donne Prix, PER, Div de {ticker}.", "Analyste", [analyse_bourse_tool], crew_models)
    if not res_fin: return None # Si échec total, on arrête cette action
    dossier += f"FINANCE: {res_fin}\n"
    
    # Etape 2 : Sentiment
    if progress_callback: progress_callback(f"🔎 {ticker} : Sentiment web...")
    res_soc = execute_step_smart("Sentiment", f"Avis web sur {ticker}.", "Trader", [recherche_web_tool], crew_models)
    if not res_soc: 
        res_soc = "Non disponible" # On continue même si le sentiment plante (optionnel)
    dossier += f"SENTIMENT: {res_soc}\n"
    
    # Etape 3 : Conclusion
    if progress_callback: progress_callback(f"🔎 {ticker} : Verdict...")
    res_con = execute_step_smart("Notation", f"Analyse le dossier pour {ticker}. Donne une note sur 10 (ex: 7/10) et une phrase de conseil PEA.", "Conseiller", [], crew_models, dossier)
    
    # Extraction de la note pour le tri
    score = 0
    match = re.search(r"(\d+)/10", str(res_con))
    if match: score = int(match.group(1))
    
    return {"Action": ticker, "Score": score, "Note": f"{score}/10", "Avis": res_con}

# --- 8. INTERFACE ONGLETS ---
tab1, tab2 = st.tabs(["🔍 Analyse Solo", "🎯 Chasseur de Pépites"])

# --- TAB 1 : SOLO ---
with tab1:
    ticker_input = st.text_input("Action à analyser", "TTE.PA", key="input_solo")
    if st.button("Lancer Solo 🚀", key="btn_solo"):
        if not api_key: st.error("Clé manquante")
        else:
            with st.status("Analyse en cours...", expanded=True):
                res = analyze_one_stock(ticker_input, st.write)
                if res:
                    st.success("Terminé !")
                    st.markdown(f"### Note : {res['Note']}")
                    st.info(res['Avis'])

# --- TAB 2 : CHASSEUR ---
with tab2:
    st.info("L'IA va scanner le marché pour toi.")
    
    strategy_name = st.selectbox("Stratégie :", list(HUNTING_STRATEGIES.keys()), key="select_strat")
    
    if st.button("Lancer la Chasse 🦅", key="btn_hunt"):
        if not api_key:
            st.error("Clé manquante")
        else:
            with st.status("📡 Chasse en cours (Patience, c'est du travail de pro)...", expanded=True) as status:
                
                # 1. RECHERCHE
                st.write("🧠 Le stratège cherche des cibles...")
                tickers_found = hunt_tickers(HUNTING_STRATEGIES[strategy_name])
                
                if not tickers_found:
                    status.update(label="Rien trouvé.", state="error")
                    st.error("Le stratège n'a pas pu identifier de cibles valides.")
                else:
                    st.success(f"Cibles identifiées : {', '.join(tickers_found)}")
                    st.write("---")
                    
                    results_data = []
                    prog_bar = st.progress(0)
                    table_spot = st.empty()
                    
                    # 2. ANALYSE EN BOUCLE
                    for i, ticker in enumerate(tickers_found):
                        # Pause préventive entre les actions pour recharger les quotas
                        if i > 0:
                            st.write(f"⏳ Pause tactique (5s) avant {ticker}...")
                            time.sleep(5) 
                            
                        res = analyze_one_stock(ticker, st.write)
                        
                        if res:
                            results_data.append(res)
                            # Tableau trié en temps réel
                            df = pd.DataFrame(results_data).sort_values(by="Score", ascending=False)
                            
                            table_spot.dataframe(
                                df[["Action", "Note", "Avis"]], 
                                use_container_width=True,
                                column_config={"Avis": st.column_config.TextColumn("Verdict IA", width="large")}
                            )
                        
                        prog_bar.progress((i + 1) / len(tickers_found))

                    status.update(label="Chasse terminée !", state="complete", expanded=False)
                    st.balloons()