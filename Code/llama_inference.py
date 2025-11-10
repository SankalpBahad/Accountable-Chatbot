from tqdm import tqdm 
import os
import requests
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
tokenizer.pad_token_id = tokenizer.eos_token_id
model.to(device)

embedding_model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")

# --- Utility Function ---
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# --- Query Improvement ---
def improve_query_hf(query):
    prompt = f"""Rewrite the following legal query to maximize retrieval accuracy from the Indian Penal Code (IPC) using cosine similarity.
    Expand the query by incorporating relevant legal terms, synonyms, related offenses, and broader legal contexts.
    Ensure the language is precise and comprehensive to improve matching with relevant laws.
    Do not include specific article numbers or factual information—focus solely on enhancing the query's expressiveness. Increase the number of keywords so that cosine similarity retrieval is aided
    Provide only the improved query, nothing else.

    Query:
    {query}
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, tokenizer=tokenizer, max_length=150)
    improved_query = tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_text(improved_query[len(prompt):])

# --- Pinecone Search ---
def search_pinecone(query, index_name):
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()
    pc = Pinecone("pinecone_id")  # Replace securely
    index = pc.Index(index_name)
    search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

    results_list = []
    for match in search_results["matches"]:
        results_list.append({
            "Offense": match["metadata"].get("Offense", "N/A"),
            "Description": match["metadata"].get("Description", "N/A"),
            "Punishment": match["metadata"].get("Punishment", "N/A")
        })
    return results_list

# --- Scrapers ---
def scrape_livelaw_article(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        article_body = soup.find('div', class_='_s30J')
        if article_body:
            content = "\n".join(p.strip() for p in article_body.find_all(text=True, recursive=False))
            return content
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
    return None

def scrape_toi_article(url, results_list):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        divs = soup.find_all('div', class_='crmK8')

        for div in divs:
            a_tags = div.find_all("a", href=True)
            for a in a_tags[:5]:
                href = a["href"]
                full_link = f"https://www.livelaw.in{href}" if href.startswith("/") else href
                content = scrape_livelaw_article(full_link)
                if content:
                    results_list.append({"Article Heading": a.text.strip(), "Content": content})
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")

# --- Response Generation ---
def extract_relevant_points_hf(query, results):
    prompt = f"""Given the following legal query, extract and arrange the most relevant points from the provided legal texts.
    Ensure the points are clear, structured, and directly address the query.
    Remove any redundant or unnecessary information.

    Query:
    {query}

    Legal Texts:
    {results}

    Make sure your response is in a structured format. Use proper bullet points and headings like Markdown File.
    Do add citations properly from the content given to you above. Add the citations in the end of your response.
    Do not use any of your own knowledge.
    Do not give anything other than the content.
    Do not repeat anything in the response and only give relevant content.
    Start your response as Relevant Legal Points and end with the citations and End of Response tag.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, tokenizer=tokenizer, max_length=len(prompt) + 10)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_text(response[len(prompt):])

# --- Pipeline for Single Query ---
def process_query(query):
    improved_query = improve_query_hf(query)
    if not improved_query:
        improved_query = query

    ipc_results = search_pinecone(improved_query, "ipc")
    nyaya_results = search_pinecone(improved_query, "nyaya-sanhita")
    combined_results = ipc_results + nyaya_results

    toi_url = f"https://timesofindia.indiatimes.com/topic/{query}/news"
    scrape_toi_article(toi_url, combined_results)

    final_response = extract_relevant_points_hf(query, combined_results)

    return {
        "Original Query": query,
        "Improved Query": improved_query,
        "Final Response": final_response
    }

# --- Main Execution for Multiple Queries ---
def main(queries, output_file="responses.json"):
    all_results = []
    for query in tqdm(queries):
        # print(f"\nProcessing: {query}")
        result = process_query(query)
        all_results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"\nResponses saved to {output_file}")


# --- Run ---
if __name__ == "__main__":
    queries_list = [
        "Punishment for sex with a minor",
        "Legal consequences of rape",
        "Penalty for molestation",
        "Laws against sexual harassment at the workplace",
        "Punishment for acid attacks",
        "Is marital rape a crime in India?",
        "Consequences of stalking and voyeurism",
        "Punishment for sharing obscene content online",
        "Laws on human trafficking and forced prostitution",
        "Legal consequences of dowry harassment",
        "Punishment for cyberstalking and online harassment",
        "Legal consequences of hacking",
        "Is online defamation a criminal offense?",
        "Penalty for identity theft and phishing scams",
        "Laws on sending obscene messages over the internet",
        "Punishment for spreading fake news and misinformation",
        "Can offensive social media posts lead to arrest?",
        "Legal actions against deepfake pornography",
        "Consequences of unauthorized access to personal data",
        "Punishment for cyberbullying and online blackmai",
        "Punishment for murder",
        "Difference between murder and culpable homicide",
        "Legal consequences of attempted murder",
        "Penalty for abetment of suicide",
        "Laws against honor killings",
        "Is mercy killing (euthanasia) legal in India?",
        "Punishment for dowry-related deaths",
        "Can accidental death lead to criminal charges?",
        "Legal consequences of mob lynching",
        "Laws on custodial death and police brutality",
        "Punishment for theft and burglary",
        "What is criminal breach of trust?",
        "Penalty for cheating and financial fraud",
        "Punishment for forgery and fake document creation",
        "Legal consequences of bank fraud",
        "Is cryptocurrency fraud a punishable offense?",
        "Laws on insider trading and corporate fraud",
        "Punishment for money laundering",
        "Consequences of issuing a bounced cheque",
        "Is running a pyramid scheme illegal in India?",
        "Punishment for kidnapping and abduction",
        "Laws on human trafficking",
        "Penalty for extortion and blackmail",
        "Legal consequences of organized crime activities",
        "Punishment for drug trafficking and narcotics smuggling",
        "Punishment for sedition and anti-national activities",
        "Laws on hate speech and communal violence",
        "Consequences of spreading religious disharmony",
        "Penalty for rioting and unlawful assembly",
        "Consequences of damaging public property during protests",
        "Laws on unlawful assembly and public disorder",
    ]
    main(queries_list)
