st.set_page_config(page_title="Le charte visite 🐱", layout="centered")
st.title("Le charte visite 🐱")

# Capture de l'image via la caméra
image_file = st.camera_input("Prenez une photo des cartes de visite")

# Ajout de la qualification du leads et de la note en dessous de la zone de photo
qualification = st.selectbox(
    "Qualification du leads",
    ["Smart Talk", "Incubation collective", "Incubation individuelle", "Transformation numérique"]
)

note = st.text_area("Ajouter une note", placeholder="Entrez votre note ici...")

if image_file is not None:
    st.image(image_file, caption="Carte de visite capturée", use_column_width=True)
    image_bytes = image_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"
    
    try:
        # Appel à l'API OCR de Mistral
        ocr_response = client_mistral.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "image_url", "image_url": image_data_uri}
        )
        st.subheader("Résultat brut de l'OCR :")
        st.write(ocr_response)
        
        # Extraction du texte exploitable
        ocr_text = extract_text_from_ocr_response(ocr_response)
        if not ocr_text:
            st.warning("Aucun texte exploitable extrait par l'OCR.")
        else:
            st.subheader("Texte OCR extrait :")
            st.text(ocr_text)
            
            # Création d'un thread pour la conversation avec l'assistant
            thread = client_openai.beta.threads.create()
            
            # Intégration de la qualification et de la note dans le message utilisateur
            user_message = (
                f"Qualification: {qualification}\n"
                f"Note: {note}\n\n"
                f"Voici le texte OCR extrait :\n{ocr_text}\n\n"
                "Extrais les informations suivantes : nom, prenom, entreprise. "
                "Ensuite, effectue une recherche en ligne pour obtenir un maximum d'informations sur l'intervenant et son entreprise."
            )
            client_openai.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )
            
            # Création d'un run pour que l'assistant traite le message
            run = client_openai.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
            run = wait_for_run_completion(thread.id, run.id)
            if run.status == 'failed':
                st.error(run.error)
            elif run.status == 'requires_action':
                run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls)
                run = wait_for_run_completion(thread.id, run.id)
            
            st.subheader("Résultat final de l'assistant :")
            print_messages_from_thread(thread.id)
            
    except Exception as e:
        st.error(f"Erreur lors du traitement OCR ou de l'analyse par l'assistant : {e}")
