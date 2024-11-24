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



import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmazonScraper:
    def __init__(self, chromedriver_path):
        self.service = Service(chromedriver_path)
        self.driver = None
        self.scraped_data = {}
        self.base_url = "https://sellercentral.amazon.in/spec/productcompliance/form?clientName=spec_web"
        
    def initialize_driver(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--start-maximized')
        chrome_options.add_argument('--disable-notifications')
        chrome_options.add_argument('--disable-popup-blocking')
        self.driver = webdriver.Chrome(service=self.service, options=chrome_options)
        
    def reload_page(self):
        """Reload the page and wait for it to be ready"""
        try:
            logger.info("Reloading page...")
            self.driver.get(self.base_url)
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(3)  # Additional wait to ensure page is fully loaded
            return True
        except Exception as e:
            logger.error(f"Error reloading page: {str(e)}")
            return False
            
    def wait_and_find_element(self, by, value, timeout=15, clickable=False):
        try:
            if clickable:
                return WebDriverWait(self.driver, timeout).until(
                    EC.element_to_be_clickable((by, value))
                )
            return WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
        except TimeoutException:
            logger.error(f"Timeout waiting for element: {value}")
            raise
            
    def scrape_single_option(self, option_index):
        """Scrape a single option with retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} for option {option_index}")
                
                # Make sure we're on the correct page
                if attempt > 0:
                    if not self.reload_page():
                        continue
                
                # Wait for and click the dropdown
                dropdown = self.wait_and_find_element(
                    By.XPATH, 
                    "//div[@class=' css-1x9fncp-indicatorContainer']", 
                    clickable=True
                )
                dropdown.click()
                time.sleep(2)
                
                # Find and interact with input field
                input_field = self.wait_and_find_element(By.ID, "react-select-2-input")
                input_field.send_keys("Field")
                # Press down key the required number of times
                for _ in range(option_index):
                    input_field.send_keys(Keys.DOWN)
                    time.sleep(0.5)
                
                input_field.send_keys(Keys.ENTER)
                time.sleep(2)
                
                # Get selected field text
                selected_field = self.wait_and_find_element(
                    By.CSS_SELECTOR, 
                    ".css-1uccc91-singleValue span"
                ).text
                logger.info(f"Selected field: {selected_field}")
                
                # Click search and get content
                search_button = self.wait_and_find_element(
                    By.ID, 
                    "product_search_btn",
                    clickable=True
                )
                search_button.click()
                time.sleep(3)
                
                content_element = self.wait_and_find_element(
                    By.CSS_SELECTOR, 
                    ".react-pdf__Page__textContent"
                )
                
                # Store the data
                self.scraped_data[selected_field] = content_element.text
                
                # Success - return True
                return True
                
            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1} for option {option_index}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed all attempts for option {option_index}")
                    return False
                time.sleep(2)  # Wait before retry
                
    def run_scraper(self, num_options=10):
        """Main scraping function"""
        try:
            self.initialize_driver()
            logger.info("Starting scraping process...")
            
            # Load initial page
            self.reload_page()
            
            # Process each option
            for i in range(num_options):
                logger.info(f"Processing option {i + 1}/{num_options}")
                success = self.scrape_single_option(i)
                
                if success:
                    logger.info(f"Successfully scraped option {i + 1}")
                    # Save after each successful scrape
                    self.save_data()
                else:
                    logger.warning(f"Failed to scrape option {i + 1}")
                
                # Reload page for next iteration
                self.reload_page()
                
        except Exception as e:
            logger.error(f"Fatal error during scraping: {str(e)}")
            raise
        finally:
            self.save_data()
            if self.driver:
                self.driver.quit()
                logger.info("Browser closed")
                
    def save_data(self):
        try:
            with open("scraped_data.json", "w", encoding="utf-8") as json_file:
                json.dump(self.scraped_data, json_file, ensure_ascii=False, indent=4)
            logger.info("Data saved successfully to scraped_data.json")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")


# scraper for amazon compliance as it needs
scraper = AmazonScraper(r'chromedriver-win64\chromedriver.exe')
scraper.run_scraper(10)


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
download_folder = "./pdfs"

all_documents=[]

# sample rag on two small documents 
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


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3,google_api_key=os.getenv("GOOGLE_API_KEY"))


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

