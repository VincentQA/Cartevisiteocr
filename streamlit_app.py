import os
import time
from mistralai import Mistral

# Récupérer la clé API depuis les variables d'environnement
api_key = os.environ["MISTRAL_API_KEY"]

# Spécifier le modèle
model = "mistral-small-latest"

# Initialiser le client Mistral
client = Mistral(api_key=api_key)

# Fonction de réessai pour appeler l'API de chat
def call_mistral_chat(client, model, messages, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.complete(model=model, messages=messages)
            return response
        except Exception as e:
            # Vérifier si l'erreur correspond à une erreur 429 (rate limit exceeded)
            if "429" in str(e):
                wait_time = 2 ** retries  # Exponential backoff: 1, 2, 4 secondes...
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1
            else:
                raise e
    raise Exception("Max retries exceeded for Mistral chat API")

# Définir les messages pour le chat
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "what is the last sentence in the document"
            },
            {
                "type": "document_url",
                "document_url": "https://arxiv.org/pdf/1805.04770"
            }
        ]
    }
]

# Appel à la fonction avec gestion des réessais
chat_response = call_mistral_chat(client, model, messages)

# Afficher la réponse
print(chat_response.choices[0].message.content)
