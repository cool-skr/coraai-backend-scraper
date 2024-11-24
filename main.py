import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import time
import concurrent.futures
from urllib.robotparser import RobotFileParser

from langchain_community.document_loaders import PyPDFLoader , TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# import clickhouse_connect

import os
from dotenv import load_dotenv

def can_crawl(url):
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    rp.read()
    return rp.can_fetch("*", url)

def is_valid_url(url, root_domain):
    parsed = urlparse(url)
    return parsed.netloc == root_domain and parsed.scheme in ["http", "https"]

def download_pdf(url, download_folder):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filename = os.path.join(download_folder, os.path.basename(url))
        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return f"Downloaded: {filename}"
    return f"Failed to download: {url}"

def crawl_website(root_url, download_folder, max_threads=5, delay=1):
    visited = set()
    queue = [root_url]
    root_domain = urlparse(root_url).netloc
    os.makedirs(download_folder, exist_ok=True)
    
    if not can_crawl(root_url):
        print("Crawling is disallowed by robots.txt.")
        return
    
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        while queue:
            current_url = queue.pop(0)
            if current_url in visited:
                continue
            visited.add(current_url)
            print(f"\nVisiting: {current_url}")
            try:
                response = requests.get(current_url)
                if response.status_code != 200:
                    continue
                soup = BeautifulSoup(response.content, "html.parser")
                for a_tag in soup.find_all("a", href=True):
                    new_url = urljoin(current_url, a_tag["href"])
                    if new_url not in visited and is_valid_url(new_url, root_domain):
                        queue.append(new_url)
                pdf_links = [urljoin(current_url, link["href"]) for link in soup.find_all("a", href=True) if link["href"].endswith(".pdf")]
                futures = [executor.submit(download_pdf, pdf_url, download_folder) for pdf_url in pdf_links]
                for future in concurrent.futures.as_completed(futures):
                    print(future.result())
                text_content = soup.get_text(separator=" ", strip=True)  
                print("\nText Content:\n")
                print(text_content)
            except Exception as e:
                print(f"Error processing {current_url}: {e}")
            time.sleep(delay)
            tqdm(total=len(visited), desc="Crawling Progress", ncols=100).update(len(visited))

def scrape_page(url, download_folder):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to access {url}: {response.status_code}")
            return
        soup = BeautifulSoup(response.content, "html.parser")
        text_content = soup.get_text(separator=" ", strip=True)
        print("\nText Content:\n")
        print(text_content)
        os.makedirs(download_folder, exist_ok=True)
        pdf_links = [urljoin(url, link["href"]) for link in soup.find_all("a", href=True) if link["href"].endswith(".pdf")]
        if pdf_links:
            print("\nDownloading PDFs...\n")
            for pdf_url in pdf_links:
                print(download_pdf(pdf_url, download_folder))
        else:
            print("\nNo PDFs found on this page.\n")
    except Exception as e:
        print(f"Error processing {url}: {e}")


root_website = "https://example.com"
root_website="https://www.indiantradeportal.in/vs.jsp?lang=0&id=0,1,30622,30624"
root_website="https://www.dgft.gov.in/CP/?opt=RoDTEP"
# root_website="https://sellercentral.amazon.in/spec/productcompliance/form?clientName=spec_web"
download_folder = "./pdfs"
# crawl_website(root_website, download_folder)
# scrape_page(root_website,download_folder)





##### code here starts for rag


all_documents=[]
# loader = PyPDFLoader(r"E:\work\scrape\pdfs\RoDTEP.pdf")

loader = PyPDFLoader(r"E:\work\scrape\pdfs\Citizen Charter.pdf")
data = loader.load()
all_documents.extend(data)
loader = TextLoader(r"E:\work\scrape\texts\amazon_compliance.txt")
data = loader.load()
all_documents.extend(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=350)
docs = text_splitter.split_documents(all_documents)

print("Total number of Chunks: ", len(docs)) 

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))




vectorstoredb = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstoredb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# retrieved_docs = retriever.invoke("achieving a target of")
# print(len(retrieved_docs))
# print("Retrieved text: ",retrieved_docs[0].page_content) 



llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3,google_api_key=os.getenv("GOOGLE_API_KEY"))


# Define a system prompt
# system_prompt = (
#    "You are a government expert. Provide clear, concise answers based on the provided context. "
#     "If the information is not found in the context, state that the answer is unavailable. "
#     "Use a maximum of three sentences."
#     "\n\n"
#     "{context}"
# )


system_prompt = (
   "You are a government expert in import and export regulations. Provide clear, concise answers based on the provided context. "
    "If the information is not found in the context, state that the answer is unavailable. "
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, chain)

response = rag_chain.invoke({"input": "What is required in the certification for eyewear export ?"})
print("Response1 "+response['answer'])
response = rag_chain.invoke({"input": "What is the governments goal to reach 2 trillion about?"})
print("Response2 "+response['answer'])

