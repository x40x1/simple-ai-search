import requests
import tkinter as tk
from tkinter import scrolledtext
from bs4 import BeautifulSoup
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import dotenv
from googlesearch import search
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
import threading

nltk.data.path.append('C:\\Users\\alexa\\AppData\\Roaming\\nltk_data')

endpoint = "https://models.inference.ai.azure.com"
model_name = "Mistral-large-2407"
token = dotenv.get_key(".env", "GITHUB_TOKEN")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

def search_web(query):
    # Perform a Google search and return the top 5 results
    results = []
    for result in search(query, num_results=5):
        results.append(result)
    return results

def summarize_url(url):
    # Fetch the content of the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Parse the HTML content and summarize it
    parser = HtmlParser.from_string(str(soup), url, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    
    return " ".join([str(sentence) for sentence in summary])

# Funktion zur Verarbeitung der Anfrage
def process_query(query, text_widget):
    web_results = search_web(query)
    summaries = [summarize_url(url) for url in web_results]
    summaries_str = "\n".join(summaries)
    
    response = client.complete(
        messages=[
            SystemMessage(content="Du bist eine hilfreiche Internet-Suchmaschine, der Suchergebnisse in Echtzeit liefern kann. Antworte immer auf deutsch und in einem kurzen Fließtext nicht in Stichpunkten. Du musst immer anhand der Suchergebnisse auf die Anfrage antworten.Sag niemal, dass du keine Aktuellen Daten hast, sondern gib immer eine Antwort, die auf den Suchergebnissen basiert."),
            UserMessage(content=f"Suchergebnisse für '{query}':\n{summaries_str}"),
        ],
        model=model_name,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0
    )
    
    answer = response.choices[0].message.content
    text_widget.insert(tk.END, "Antwort: " + answer + "\n\n\n\n\n\n\n\n")

# Funktion zum Starten des Anfrage-Threads
def start_query():
    query = query_entry.get()
    threading.Thread(target=process_query, args=(query, text_widget)).start()

# GUI erstellen
root = tk.Tk()
root.title("Suchmaschine")

# Grid-Layout verwenden
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=10)
root.grid_columnconfigure(0, weight=1)

query_entry = tk.Entry(root, width=50)
query_entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

search_button = tk.Button(root, text="Suchen", command=start_query)
search_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

text_widget = scrolledtext.ScrolledText(root, width=60, height=20)
text_widget.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

root.mainloop()