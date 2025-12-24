import streamlit as st
import os
import time
import google.generativeai as genai
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS
import yfinance as yf

# --- CONFIGURATION ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

st.set_page_config(page_title="Agent PEA Filtré", page_icon="🛡️", layout="wide")

st.title("🛡️ Assistant PEA (Filtre Intelligent)")
st.markdown("""
Système auto-adaptatif avec **exclusion des modèles incompatibles** (Audio/Vision).
Seuls les modèles textuels stables sont utilisés.
""")

# --- FONCTION DE DECOUVERTE BLINDÉE ---
def get_active_models(api_key):
    """
    Récupère la liste des modèles ET exclut ceux qui ne font pas de texte pur.
    """
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        
        valid_models = []
        for m in models:
            name = m.name.lower()
            methods = m.supported_generation_methods
            
            # CRITÈRES D'EXCLUSION STRICTS
            if 'generateContent' not in methods: continue # Doit pouvoir générer du contenu
            if 'tts' in name: continue           # Pas de modèles Audio (Text-to-Speech)
            if 'vision' in name: continue        # Pas de modèles Vision purs
            if 'embedding' in name: continue     # Pas de modèles d'embedding
            if 'geek' in name or 'gecko' in name: continue # Modèles trop petits
            
            valid_models.append(m.name)
        
        # TRI PAR PERFORMANCE (Flash Stable > Flash Exp > Pro)
        sorted_models = sorted(valid_models, key=lambda x: (
            0 if "gemini-2.0-flash" in x and "exp" not in x else  # Le roi : 2.0 Flash Stable
            1 if "gemini-1.5-flash" in x and "8b" not in x else   # Le vice-roi : 1.5 Flash Stable
            2 if "flash" in x else                                # Les autres Flash
            3 if "pro" in x else                                  # Les Pro (plus lents)
            4                                                     # Le reste
        ))
        
        # Format CrewAI
        crew_models = [m.replace("models/", "gemini/") for m in sorted_models]
        return crew_models
    except Exception as e:
        return []

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Ta clé Google API", type="password")
    
    if api_key:
        available_models = get_active_models(api_key)
        if available_models:
            st.success(f"✅ {len(available_models)} modèles TEXTE valides !")
            with st.expander("Voir la liste filtrée"):
                for i, m in enumerate(available_models):
                    st.caption(f"{i+1}. {m}")
        else:
            st.error("Aucun modèle valide trouvé.")

# --- OUTILS ---
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

# --- EXECUTEUR D'ÉTAPE ---
def execute_step_smart(step_name, task_description, agent_role, agent_tools, model_list, context_data=""):
    
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key

    for model_name in model_list:
        try:
            clean_name = model_name.replace("gemini/", "")
            
            my_llm = LLM(model=model_name, api_key=api_key, temperature=0.1)

            agent = Agent(
                role=agent_role,
                goal="Tâche unique",
                backstory="Expert.",
                verbose=True,
                allow_delegation=False,
                llm=my_llm,
                tools=agent_tools,
                max_rpm=10
            )

            full_desc = task_description
            if context_data:
                full_desc += f"\nINFO CONTEXTE :\n{context_data}"

            task = Task(description=full_desc, expected_output="Réponse courte.", agent=agent)

            crew = Crew(agents=[agent], tasks=[task], verbose=True)
            result = crew.kickoff()
            
            st.toast(f"✅ Étape '{step_name}' réussie ({clean_name})", icon="🎉")
            return str(result)

        except Exception as e:
            error_str = str(e)
            # Gestion des erreurs typiques
            if "400" in error_str and "modalities" in error_str:
                # C'est l'erreur TTS ! On passe.
                continue
            if "404" in error_str: continue 
            elif "429" in error_str or "Quota" in error_str or "ResourceExhausted" in error_str:
                st.toast(f"⚠️ {clean_name} épuisé. Suivant...", icon="🔀")
                time.sleep(1)
                continue
            else:
                # On log l'erreur mais on essaie quand même le suivant au cas où
                st.warning(f"Erreur sur {clean_name} : {e}")
                continue

    st.error(f"❌ Échec de l'étape '{step_name}' sur tous les modèles.")
    return None

# --- ORCHESTRATION ---
def run_full_analysis(ticker):
    model_list = get_active_models(api_key)
    if not model_list:
        st.error("Aucun modèle disponible.")
        return None

    dossier = ""
    
    # ETAPE 1
    with st.spinner("📊 Analyse Financière..."):
        res_finance = execute_step_smart(
            "Finance", f"Donne Prix, PER et Dividende pour {ticker}.",
            "Analyste", [analyse_bourse_tool], model_list
        )
        if not res_finance: return None
        dossier += f"FINANCE:\n{res_finance}\n\n"
        st.info(f"💰 Données : {res_finance}")

    # ETAPE 2
    with st.spinner("🌍 Analyse Sentiment..."):
        res_social = execute_step_smart(
            "Sentiment", f"Cherche sentiment web sur {ticker}.",
            "Trader", [recherche_web_tool], model_list
        )
        if not res_social: return None
        dossier += f"SENTIMENT:\n{res_social}\n\n"

    # ETAPE 3
    with st.spinner("🧠 Synthèse..."):
        res_final = execute_step_smart(
            "Conclusion", f"Conseil PEA pour {ticker} (Achat/Vente).",
            "Conseiller", [], model_list, context_data=dossier
        )
        return res_final

# --- EXECUTION ---
ticker = st.text_input("Action (ex: TTE.PA)", "TTE.PA")

if st.button("Lancer l'analyse 🚀"):
    if not api_key:
        st.error("Clé manquante !")
    else:
        final_report = run_full_analysis(ticker)
        if final_report:
            st.divider()
            st.success("Terminé !")
            st.markdown("### 🏆 Rapport Final")
            st.markdown(final_report)