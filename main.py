from dataclasses import dataclass
import json
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from googleapiclient.discovery import build
from google.oauth2 import service_account

with open('openai-api-key.txt', 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()
os.environ["NEO4J_URI"] = 'neo4j://localhost:7687'
os.environ["NEO4J_USERNAME"] = 'neo4j'
os.environ["NEO4J_PASSWORD"] = 'qlonolink'

@dataclass
class GoolgeDriveFile:
    id: str
    name: str
    content: str

def get_google_drive_files() -> List[GoolgeDriveFile]:
    current_file_path = os.path.abspath(__file__)
    google_service_account_json_file_path = os.path.join(
        os.path.dirname(current_file_path), "google-service-account.json")
    if not os.path.exists(google_service_account_json_file_path):
        raise ValueError(
            "google-service-account.json is required in the same directory as the script")
    
    with open(google_service_account_json_file_path, "r") as f:
        google_service_account_json = f.read()

    SCOPES = ['https://www.googleapis.com/auth/drive']
    google_service_account = json.loads(google_service_account_json)
    creds = service_account.Credentials.from_service_account_info(
        google_service_account, scopes=SCOPES)

    googledrive_service = build('drive', 'v3', credentials=creds)
    results = googledrive_service.files().list(
        q="mimeType='application/vnd.google-apps.document'").execute()
    googledrive_files = results.get('files', [])
    if not googledrive_files:
        print('No changes found.')
        return []

    print('Files count:', len(googledrive_files))

    result: List[GoolgeDriveFile] = []
    for file_index, googledrive_file in enumerate(googledrive_files):
        try:
            print(f"Processing file {file_index + 1}/{len(googledrive_files)}")
            file_id = googledrive_file.get('id')
            if not file_id:
                print('File ID not found for file:', googledrive_file)
                continue
            file_name = googledrive_file.get('name')
            if not file_name:
                print('File name not found for file:', googledrive_file)
                continue
            content = googledrive_service.files().export(fileId=file_id, mimeType='text/plain').execute().decode('utf-8')
            print(file_id, file_name, content[:100])
            if not content:
                print('Content not found for file:', googledrive_file)
                continue
            if not isinstance(content, str):
                print('Content is not string for file:', googledrive_file)
                continue
            result.append(GoolgeDriveFile(
                id=file_id,
                name=file_name,
                content=content
            ))
        except Exception as e:
            print(e)
            continue
    return result

def main() -> None:
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    llm_transformer = LLMGraphTransformer(llm=llm)
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    graph = Neo4jGraph()

    google_drive_files = get_google_drive_files()

    for i, google_drive_file in enumerate(google_drive_files):
        print(google_drive_file, f"{i + 1}/{len(google_drive_files)}")
        split_texts = text_splitter.split_text(google_drive_file.content)
        graph_documents = llm_transformer.convert_to_graph_documents([Document(page_content=text) for text in split_texts])
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )

if __name__ == '__main__':
    main()
