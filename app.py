import streamlit as st
import os
import time
import re
import google.generativeai as genai
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS
import yfinance as yf

# --- 1. CONFIGURATION SYSTEME ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

# --- 2. INTERFACE ---
st.set_page_config(page_title="Agent PEA Invincible", page_icon="🛡️", layout="wide")

st.title("🛡️ Assistant PEA (Invincible)")
st.markdown("""
Système à **haute résilience**. Si tous les quotas gratuits sont épuisés, 
l'agent se met en veille le temps nécessaire et **reprend automatiquement le travail**.
""")

# --- 3. FONCTIONS UTILITAIRES ---
def get_active_models(api_key):
    """Découvre et trie les modèles."""
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        valid_models = []
        for m in models:
            name = m.name.lower()
            if 'generateContent' not in m.supported_generation_methods: continue
            if any(x in name for x in ['tts', 'vision', 'embedding', 'geek', 'gecko']): continue
            valid_models.append(m.name)
        
        # Tri : Flash 2.0 > Flash 1.5 > Le reste
        return sorted(valid_models, key=lambda x: (
            0 if "gemini-2.0-flash" in x and "exp" not in x else
            1 if "gemini-1.5-flash" in x and "8b" not in x else
            2 if "flash" in x else 3
        ))
    except: return []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Ta clé Google API", type="password")
    
    if api_key:
        models = get_active_models(api_key)
        if models:
            st.success(f"✅ {len(models)} modèles prêts au combat !")
            # Transformation pour CrewAI
            crew_models = [m.replace("models/", "gemini/") for m in models]
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
    except: return "Erreur recherche."

@tool("Bourse Yahoo")
def analyse_bourse_tool(ticker: str):
    """Données Bourse."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return str({
            "Prix": info.get('currentPrice'),
            "PER": info.get('forwardPE'),
            "Div": info.get('dividendYield')
        })
    except: return "Erreur Yahoo."

# --- 6. LE CŒUR DU SYSTÈME : EXECUTEUR AVEC ATTENTE INTELLIGENTE ---
def execute_step_smart(step_name, task_description, agent_role, agent_tools, model_list, context_data=""):
    
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key

    # Paramètres de persévérance
    max_global_retries = 3 # Combien de fois on accepte d'attendre si tout est vide
    current_retry = 0

    while current_retry <= max_global_retries:
        
        retry_delays = [] # Pour stocker les temps d'attente demandés par Google
        
        # 1. On essaie chaque modèle de la liste
        for model_name in model_list:
            clean_name = model_name.replace("gemini/", "")
            
            try:
                # Création agent
                my_llm = LLM(model=model_name, api_key=api_key, temperature=0.1)
                agent = Agent(
                    role=agent_role, goal="Tâche", backstory="Expert.", verbose=True,
                    allow_delegation=False, llm=my_llm, tools=agent_tools, max_rpm=10
                )
                
                # Création tâche
                desc = task_description + (f"\nCONTEXTE:\n{context_data}" if context_data else "")
                task = Task(description=desc, expected_output="Réponse courte.", agent=agent)
                
                # Exécution
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                result = crew.kickoff()
                
                # SUCCÈS !
                st.toast(f"✅ Étape '{step_name}' réussie ({clean_name})", icon="🎉")
                return str(result)

            except Exception as e:
                error_str = str(e)
                
                # Gestion fine des erreurs
                if "404" in error_str or "400" in error_str: 
                    continue # Modèle cassé, on passe
                
                if "429" in error_str or "Quota" in error_str or "ResourceExhausted" in error_str:
                    # Extraction du temps d'attente
                    match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                    wait_time = float(match.group(1)) if match else 60.0
                    retry_delays.append(wait_time)
                    
                    st.toast(f"⚠️ {clean_name} vide (Reset: {int(wait_time)}s). Suivant...", icon="⏭️")
                    time.sleep(1) # Petite pause rapide pour changer de modèle
                    continue
                
                # Autre erreur inconnue
                st.warning(f"Erreur sur {clean_name} : {e}")

        # 2. Si on arrive ici, c'est que TOUS les modèles ont échoué
        if not retry_delays:
            st.error("❌ Échec total (Erreurs techniques non liées aux quotas).")
            return None
            
        # 3. Stratégie d'attente intelligente
        if current_retry < max_global_retries:
            # On prend le délai le plus court parmi tous les échecs + 2 secondes de sécurité
            shortest_wait = min(retry_delays) + 2
            
            # Affichage du compte à rebours bloquant
            alert_box = st.warning(f"🛑 Tous les agents sont épuisés. Pause tactique de {int(shortest_wait)}s...")
            progress_bar = st.progress(0)
            
            for i in range(int(shortest_wait)):
                time_left = int(shortest_wait) - i
                alert_box.warning(f"⏳ Recharge des batteries... Reprise dans **{time_left}s**")
                progress_bar.progress((i + 1) / int(shortest_wait))
                time.sleep(1)
            
            # Nettoyage et reprise
            alert_box.empty()
            progress_bar.empty()
            st.toast("🔋 Énergie récupérée ! Nouvelle tentative...", icon="🔄")
            current_retry += 1
            # La boucle while va recommencer au début de la liste des modèles !
            
        else:
            st.error("❌ Abandon après trop de tentatives d'attente.")
            return None

# --- 7. ORCHESTRATION ---
def run_full_analysis(ticker):
    if not crew_models:
        st.error("Pas de modèles.")
        return None

    dossier = ""
    
    # Etape 1
    with st.spinner("📊 Analyse Financière..."):
        res = execute_step_smart("Finance", f"Donne Prix, PER, Div pour {ticker}.", "Analyste", [analyse_bourse_tool], crew_models)
        if not res: return None
        dossier += f"FINANCE: {res}\n"
        st.info(f"💰 Données : {res}")

    # Etape 2
    with st.spinner("🌍 Analyse Sentiment..."):
        res = execute_step_smart("Sentiment", f"Avis web sur {ticker}.", "Trader", [recherche_web_tool], crew_models)
        if not res: return None
        dossier += f"SENTIMENT: {res}\n"

    # Etape 3
    with st.spinner("🧠 Synthèse..."):
        res = execute_step_smart("Conclusion", f"Conseil PEA {ticker}.", "Conseiller", [], crew_models, dossier)
        return res

# --- 8. UI ---
ticker = st.text_input("Action", "TTE.PA")
if st.button("Lancer 🚀"):
    if api_key:
        final = run_full_analysis(ticker)
        if final:
            st.divider()
            st.markdown("### 🏆 Résultat Final")
            st.markdown(final)
    else:
        st.error("Clé manquante")