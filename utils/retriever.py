import os
import json

import vertexai
from google.cloud import storage

from langchain.globals import set_debug
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_community import BigQueryVectorStore, VertexFSVectorStore
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings

PROJECT_ID = "ck-vertex"
LOCATION = "us-central1"
DATASET = "vector_store"
TABLE = "usv_tire_warranty"
LOCATION = "us-central1"
GCS_BUCKET = "ck-usv-tire-warranty"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Initialize the embedding model - Google Gecko
embedding_model = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=PROJECT_ID
)

# Initialize the vector store
bq_store = BigQueryVectorStore(
    project_id=PROJECT_ID,
    location=LOCATION,
    dataset_name=DATASET,
    table_name=TABLE,
    embedding=embedding_model,
)

def ingest_pdfs():

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)

    all_documents = []  # Store all loaded documents

    blobs = bucket.list_blobs()
    for blob in blobs:
        if blob.name.endswith(".pdf"):
            # Download the blob to a temporary file
            temp_file_path = f"/tmp/{blob.name}"
            blob.download_to_filename(temp_file_path)

            # Load the PDF
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            # Add metadata to the documents
            for document in documents:
                doc_md = document.metadata
                document_name = doc_md["source"].split("/")[-1]
                doc_source_prefix = "/".join(GCS_BUCKET.split("/")[:3])
                doc_source_suffix = "/".join(doc_md["source"].split("/")[4:-1])
                source = f"{doc_source_prefix}/{doc_source_suffix}"
                document.metadata = {"source": source, "document_name": document_name}

            all_documents.extend(documents)

            # Delete the temporary file
            os.remove(temp_file_path)

    # Split all documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    doc_splits = text_splitter.split_documents(all_documents)

    # Add chunk number to metadata
    for idx, split in enumerate(doc_splits):
        split.metadata["chunk"] = idx

    print(f"# of documents = {len(doc_splits)}")

    doc_ids = bq_store.add_documents(doc_splits)
    print(f"Doc IDs: {doc_ids}")

def similarity_search(prompt):

    response = bq_store.similarity_search(prompt)
    structured_data = []

    for doc in response:
        structured_data.append({
            "metadata": {
                "doc_id": doc.metadata["doc_id"],
                "source": doc.metadata["source"],
                "document_name": doc.metadata["document_name"],
                "chunk": doc.metadata["chunk"],
                "score": doc.metadata["score"]
            },
            "page_content": doc.page_content
        })

    return json.dumps(structured_data, indent=4)

#print(ingest_pdfs())