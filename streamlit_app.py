import streamlit as st
import os
import base64
from mistralai import Mistral

# Fonction d'extraction via l'IA (pour une première extraction automatique)
def extract_business_card_data_via_ai(ocr_text, client, model="mistral-small-latest"):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extrais les informations de la carte de visite suivantes. Donne-moi le nom, l'email, le téléphone et, si possible, le nom de l'entreprise."
                },
                {
                    "type": "text",
                    "text": ocr_text
                }
            ]
        }
    ]
    response = client.chat.complete(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

# Configuration de la page
st.set_page_config(page_title="Le charte visite 🐱", layout="centered")
st.title("Le charte visite 🐱")

# Capture de l'image via la caméra
image_file = st.camera_input("Prenez une photo des cartes de visite")

# Vérification si une image a été capturée
if image_file is not None:
    st.image(image_file, caption="Carte de visite capturée", use_column_width=True)
    
    # Encodage de l'image en base64 pour création d'une data URI
    image_bytes = image_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"
    
    # Récupération de la clé API Mistral
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("Clé API Mistral non trouvée. Veuillez définir MISTRAL_API_KEY dans vos variables d'environnement.")
    else:
        client = Mistral(api_key=api_key)
        
        try:
            # Appel à l'API OCR avec l'image encodée
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": image_data_uri
                }
            )
            st.subheader("Résultat de l'OCR :")
            st.write(ocr_response)
            
            # On suppose que le texte extrait est disponible dans la clé "text"
            ocr_text = ocr_response.get("text", "")
            if not ocr_text:
                st.warning("Aucun texte extrait par l'OCR.")
            else:
                st.subheader("Extraction initiale par l'IA :")
                initial_extraction = extract_business_card_data_via_ai(ocr_text, client)
                st.write(initial_extraction)
                
                # Initialisation de l'historique de chat dans la session
                if "chat_history" not in st.session_state:
                    # On ajoute un message système avec le contexte OCR afin de fournir un contexte à l'agent
                    st.session_state.chat_history = [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Voici le texte OCR extrait de la carte de visite :\n\n{ocr_text}"
                                }
                            ]
                        }
                    ]
                
                st.subheader("Discussion avec l'agent")
                
                # Affichage de l'historique du chat
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"**Vous :** {''.join([c.get('text', '') for c in msg['content']])}")
                    elif msg["role"] == "assistant":
                        st.markdown(f"**Agent :** {''.join([c.get('text', '') for c in msg['content']])}")
                    elif msg["role"] == "system":
                        st.markdown(f"**Contexte :** {''.join([c.get('text', '') for c in msg['content']])}")
                
                # Zone de saisie pour le message utilisateur
                user_input = st.text_input("Votre question pour l'agent", key="user_input")
                
                if st.button("Envoyer") and user_input:
                    # Ajout du message utilisateur dans l'historique
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_input
                            }
                        ]
                    })
                    
                    with st.spinner("L'agent réfléchit..."):
                        # Appel de l'agent avec l'historique complet
                        response = client.chat.complete(
                            model="mistral-small-latest",
                            messages=st.session_state.chat_history
                        )
                        agent_reply = response.choices[0].message.content
                        
                        # Ajout de la réponse de l'agent à l'historique
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": agent_reply
                                }
                            ]
                        })
                    
                    # Rafraîchissement de l'affichage du chat
                    st.experimental_rerun()
        except Exception as e:
            st.error(f"Erreur lors du traitement OCR ou de l'analyse par l'IA : {e}")
