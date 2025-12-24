import streamlit as st
import os
import time
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS
import yfinance as yf

# --- CONFIGURATION ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

st.set_page_config(page_title="Agent PEA Checkpoints", page_icon="💾", layout="wide")

st.title("💾 Assistant PEA (Sauvegarde d'Étapes)")
st.markdown("""
Ce système fonctionne par **étapes**. Si un agent échoue par manque de quota, 
un autre agent prend le relais **uniquement pour l'étape en cours**, sans tout recommencer.
""")

# --- LISTE DES MODELES (Du plus rapide au plus robuste) ---
GEMINI_MODELS = [
    "gemini/gemini-2.0-flash",
    "gemini/gemini-1.5-flash",
    "gemini/gemini-1.5-flash-8b",
    "gemini/gemini-pro"
]

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Ta clé Google API", type="password")
    st.divider()
    console_log = st.empty() # Zone de logs temps réel

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
def execute_step_smart(step_name, task_description, agent_role, agent_tools, context_data=""):
    """
    Exécute une seule tâche. Si ça plante, change de modèle et réessaie LA MÊME tâche.
    """
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key

    # On essaie les modèles un par un
    for model_name in GEMINI_MODELS:
        try:
            clean_name = model_name.replace("gemini/", "")
            console_log.info(f"🔄 Étape '{step_name}' en cours avec : **{clean_name}**")
            
            # 1. Création du cerveau
            my_llm = LLM(model=model_name, api_key=api_key, temperature=0.1)

            # 2. Création de l'agent pour cette étape spécifique
            agent = Agent(
                role=agent_role,
                goal="Exécuter la tâche demandée.",
                backstory="Expert précis.",
                verbose=True,
                allow_delegation=False,
                llm=my_llm,
                tools=agent_tools,
                max_rpm=10
            )

            # 3. Préparation de la tâche
            # On injecte le "Dossier" (context_data) directement dans la description
            full_description = task_description
            if context_data:
                full_description += f"\n\nVOICI LE DOSSIER DES ETAPES PRECEDENTES (Utilise ces infos) :\n{context_data}"

            task = Task(
                description=full_description,
                expected_output="Réponse synthétique.",
                agent=agent
            )

            # 4. Exécution
            crew = Crew(agents=[agent], tasks=[task], verbose=True)
            result = crew.kickoff()
            
            # Si on arrive ici, c'est que ça a marché !
            return str(result)

        except Exception as e:
            # Si erreur de quota, on capture et on continue la boucle (modèle suivant)
            if "429" in str(e) or "Quota" in str(e) or "ResourceExhausted" in str(e):
                st.toast(f"⚠️ {clean_name} épuisé sur l'étape '{step_name}'. Passage au suivant...", icon="🔀")
                time.sleep(2) # Petite pause respiration
                continue # On essaie le prochain modèle de la liste
            else:
                # Si c'est une autre erreur, on l'affiche
                st.error(f"Erreur technique sur {clean_name} : {e}")
                return None

    st.error("❌ Tous les agents ont échoué pour cette étape.")
    return None

# --- ORCHESTRATION DU DOSSIER ---
def run_full_analysis(ticker):
    dossier = "" # C'est ici qu'on stocke la mémoire du projet
    
    # --- ETAPE 1 : FINANCE ---
    with st.spinner("📊 Étape 1/3 : Analyse Financière..."):
        res_finance = execute_step_smart(
            step_name="Finance",
            task_description=f"Donne uniquement Prix, PER et Dividende pour {ticker}.",
            agent_role="Analyste Financier",
            agent_tools=[analyse_bourse_tool]
        )
        if not res_finance: return None
        
        # On ajoute au dossier
        dossier += f"--- DONNÉES FINANCIÈRES ---\n{res_finance}\n\n"
        st.success("✅ Données financières sécurisées !")
        with st.expander("Voir les données brutes"):
            st.write(res_finance)

    # --- ETAPE 2 : SENTIMENT ---
    with st.spinner("🌍 Étape 2/3 : Analyse Sentiment..."):
        res_social = execute_step_smart(
            step_name="Sentiment",
            task_description=f"Cherche sur le web l'avis des investisseurs sur {ticker}.",
            agent_role="Trader Web",
            agent_tools=[recherche_web_tool]
        )
        if not res_social: return None
        
        # On ajoute au dossier
        dossier += f"--- SENTIMENT SOCIAL ---\n{res_social}\n\n"
        st.success("✅ Sentiment social sécurisé !")
        with st.expander("Voir le sentiment brut"):
            st.write(res_social)

    # --- ETAPE 3 : SYNTHÈSE (Avec le dossier complet) ---
    with st.spinner("🧠 Étape 3/3 : Synthèse Finale..."):
        res_final = execute_step_smart(
            step_name="Conclusion",
            task_description=f"Agis comme un conseiller en gestion de patrimoine. Analyse le dossier ci-dessous concernant {ticker} et donne une recommandation claire (Achat/Vente/Attente) pour un PEA.",
            agent_role="Conseiller Wealth",
            agent_tools=[], # Pas besoin d'outils, il a le dossier !
            context_data=dossier # <--- ON LUI PASSE TOUT LE TRAVAIL PRÉCÉDENT
        )
        return res_final

# --- EXECUTION ---
ticker = st.text_input("Action (ex: TTE.PA)", "TTE.PA")

if st.button("Lancer l'analyse Séquentielle 🚀"):
    if not api_key:
        st.error("Clé manquante !")
    else:
        final_report = run_full_analysis(ticker)
        
        if final_report:
            st.divider()
            st.markdown("### 🏆 Rapport Final Consolidé")
            st.markdown(final_report)