import streamlit as st
import os
import google.generativeai as genai

# --- CONFIGURATION ---
# On désactive la télémétrie pour nettoyer les logs
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

st.set_page_config(page_title="Diagnostic Gemini", page_icon="🕵️", layout="centered")

st.title("🕵️ Inspecteur de Modèles Google")
st.info("Ce script va lister dans la console (logs) tous les modèles auxquels ta clé a accès.")

# --- FONCTION DE DIAGNOSTIC ---
def scan_and_print_models(api_key):
    try:
        # Configuration de l'accès Google
        genai.configure(api_key=api_key)
        
        # Marqueurs visuels pour retrouver facilement les infos dans les logs
        print("\n\n" + "█"*60)
        print("█ DEBUT DU SCAN DES MODELES DISPONIBLES")
        print("█"*60 + "\n")
        
        models = list(genai.list_models())
        text_models_count = 0
        
        if not models:
            print("❌ AUCUN MODÈLE TROUVÉ (La liste est vide).")
            return False, "Liste vide"

        for m in models:
            # On cherche les modèles qui savent générer du texte ('generateContent')
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ MODÈLE VALIDE : {m.name}")
                print(f"   Nom affiché : {m.display_name}")
                print(f"   Description : {m.description}")
                print(f"   Méthodes : {m.supported_generation_methods}")
                print("-" * 40)
                text_models_count += 1
            else:
                # On affiche quand même les autres (vision, embedding) pour info
                print(f"⚠️  MODÈLE NON-TEXTE : {m.name}")
        
        print("\n" + "█"*60)
        print(f"█ FIN DU SCAN : {text_models_count} modèles texte trouvés.")
        print("█"*60 + "\n\n")
        
        return True, text_models_count
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE : {str(e)}\n")
        return False, str(e)

# --- INTERFACE ---
api_key = st.text_input("Colle ta clé API Google ici (AIza...)", type="password")

if st.button("Lancer le Scan des Logs 🚀"):
    if not api_key:
        st.error("Il faut une clé API !")
    else:
        with st.status("Connexion à Google en cours...", expanded=True) as status:
            success, count = scan_and_print_models(api_key)
            
            if success:
                status.update(label="Scan terminé !", state="complete")
                st.success(f"✅ Succès ! {count} modèles compatibles trouvés.")
                st.markdown("""
                ### 👉 Action requise :
                1. Regarde en bas à droite de cette fenêtre.
                2. Clique sur l'onglet **'Manage App'** pour ouvrir la console noire.
                3. Copie tout ce qui se trouve entre les barres `█████`.
                4. Colle-le dans notre discussion.
                """)
            else:
                status.update(label="Erreur", state="error")
                st.error(f"Erreur technique : {count}")