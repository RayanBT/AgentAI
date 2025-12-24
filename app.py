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

st.set_page_config(page_title="Agent PEA Auto-Adaptatif", page_icon="🦎", layout="wide")

st.title("🦎 Assistant PEA (Auto-Adaptatif)")
st.markdown("""
Ce système **scanne ton compte Google** pour trouver les modèles disponibles et crée une chaîne de secours automatiquement.
Si un modèle échoue (Quota), le suivant prend le relais pour l'étape en cours.
""")

# --- FONCTION DE DECOUVERTE DES MODELES ---
def get_active_models(api_key):
    """Récupère la liste réelle des modèles disponibles pour l'utilisateur."""
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        
        # On ne garde que les modèles texte (generateContent)
        valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        # ON TRIE PAR PRIORITÉ (Flash > Pro > Experimental)
        # On veut d'abord les modèles rapides et gratuits
        sorted_models = sorted(valid_models, key=lambda x: (
            0 if "flash" in x and "2.0" in x else  # Priorité absolue : Flash 2.0
            1 if "flash" in x and "1.5" in x else  # Priorité 2 : Flash 1.5
            2 if "flash" in x else                 # Priorité 3 : Autres Flash
            3 if "pro" in x else                   # Priorité 4 : Pro
            4                                      # Le reste
        ))
        
        # Conversion au format CrewAI : on remplace 'models/' par 'gemini/'
        crew_models = [m.replace("models/", "gemini/") for m in sorted_models]
        return crew_models
    except Exception as e:
        return []

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Ta clé Google API", type="password")
    
    # Zone d'info dynamique
    if api_key:
        available_models = get_active_models(api_key)
        if available_models:
            st.success(f"✅ {len(available_models)} modèles détectés !")
            with st.expander("Voir la chaîne de secours"):
                for i, m in enumerate(available_models):
                    st.caption(f"{i+1}. {m}")
        else:
            st.error("Impossible de récupérer les modèles. Clé invalide ?")

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

# --- FONCTION INTELLIGENTE : EXECUTEUR D'ÉTAPE ---
def execute_step_smart(step_name, task_description, agent_role, agent_tools, model_list, context_data=""):
    """Exécute une tâche en essayant la liste des modèles un par un."""
    
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key

    # On boucle sur la liste réelle des modèles de l'utilisateur
    for model_name in model_list:
        try:
            # Affichage discret du modèle utilisé
            clean_name = model_name.replace("gemini/", "")
            
            # 1. Création du cerveau
            my_llm = LLM(model=model_name, api_key=api_key, temperature=0.1)

            # 2. Création de l'agent
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

            # 3. Tâche
            full_desc = task_description
            if context_data:
                full_desc += f"\nINFO CONTEXTE :\n{context_data}"

            task = Task(description=full_desc, expected_output="Réponse courte.", agent=agent)

            # 4. Exécution
            crew = Crew(agents=[agent], tasks=[task], verbose=True)
            result = crew.kickoff()
            
            # SUCCÈS !
            st.toast(f"✅ Étape '{step_name}' réussie avec {clean_name}", icon="🎉")
            return str(result)

        except Exception as e:
            error_str = str(e)
            # Gestion des erreurs
            if "404" in error_str:
                # Si le modèle n'existe pas (bizarre car on l'a scanné, mais possible), on passe
                continue 
            elif "429" in error_str or "Quota" in error_str or "ResourceExhausted" in error_str:
                st.toast(f"⚠️ {clean_name} épuisé. Bascule sur le suivant...", icon="🔀")
                time.sleep(2)
                continue
            else:
                st.error(f"Erreur technique sur {clean_name} : {e}")
                return None

    st.error("❌ Tous les modèles ont échoué.")
    return None

# --- ORCHESTRATION ---
def run_full_analysis(ticker):
    # 1. On récupère la liste fraîche des modèles
    model_list = get_active_models(api_key)
    if not model_list:
        st.error("Aucun modèle disponible.")
        return None

    dossier = ""
    
    # --- ETAPE 1 : FINANCE ---
    with st.spinner("📊 Étape 1/3 : Analyse Financière..."):
        res_finance = execute_step_smart(
            "Finance",
            f"Donne Prix, PER et Dividende pour {ticker}.",
            "Analyste",
            [analyse_bourse_tool],
            model_list
        )
        if not res_finance: return None
        dossier += f"FINANCE:\n{res_finance}\n\n"
        st.info(f"💰 Données : {res_finance}")

    # --- ETAPE 2 : SENTIMENT ---
    with st.spinner("🌍 Étape 2/3 : Analyse Sentiment..."):
        res_social = execute_step_smart(
            "Sentiment",
            f"Cherche sentiment web sur {ticker}.",
            "Trader",
            [recherche_web_tool],
            model_list
        )
        if not res_social: return None
        dossier += f"SENTIMENT:\n{res_social}\n\n"

    # --- ETAPE 3 : CONCLUSION ---
    with st.spinner("🧠 Étape 3/3 : Synthèse..."):
        res_final = execute_step_smart(
            "Conclusion",
            f"Conseil PEA pour {ticker} (Achat/Vente) basé sur le dossier.",
            "Conseiller",
            [], # Pas d'outils
            model_list,
            context_data=dossier
        )
        return res_final

# --- EXECUTION ---
ticker = st.text_input("Action (ex: TTE.PA)", "TTE.PA")

if st.button("Lancer l'analyse Auto-Adaptative 🚀"):
    if not api_key:
        st.error("Clé manquante !")
    else:
        final_report = run_full_analysis(ticker)
        if final_report:
            st.divider()
            st.success("Analyse complète terminée !")
            st.markdown("### 🏆 Rapport Final")
            st.markdown(final_report)