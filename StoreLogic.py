import io
import fitz  
import docx
import ftfy  
from typing import List, Tuple, Any
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_milvus.vectorstores import Zilliz


def get_text_from_uploaded_file(uploaded_file: Any) -> Tuple[str | None, str | None]:
    """
    Reads and aggressively cleans the text content from an uploaded file.
    """
    text = ""
    err_message = None
   
    file_name = uploaded_file.name
    
    try:
        if file_name.endswith('.pdf'):
            pdf_bytes = uploaded_file.read()
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
       
        elif file_name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join(para.text for para in doc.paragraphs)
       
        elif file_name.endswith('.txt'):
            raw_bytes = uploaded_file.read()
            text = ftfy.fix_encoding(raw_bytes.decode('latin-1'))
       
        else:
            err_message = 'Unsupported file extension.'

        if text:
            text = ftfy.fix_text(text)
            
    except Exception as e:
        err_message = f"An error occurred while reading the file: {e}"
        
    return text, err_message


def semantic_chunk(text: str, embedding_model: Any) -> Tuple[List[str] | None, str | None]:
    if not text or not text.strip(): 
        return [], None
    
    try:
        text_splitter = SemanticChunker(embedding_model)
        chunks = text_splitter.split_text(text)
        return chunks, None
   
    except Exception as e:
        return None, f"An error occurred during chunking: {e}"

def initialize_vector_store(embedding_model: Any, zilliz_args: dict) -> Tuple[Any, str, str | None]:
    uri, token, collection_name = zilliz_args.get("uri"), zilliz_args.get("token"), zilliz_args.get("collection_name")
    
    if all([uri, token, collection_name]):
        try:
            vector_store = Zilliz(embedding_function=embedding_model, collection_name=collection_name, connection_args={"uri": uri, "token": token})
            return vector_store, "zilliz", "Connected to Zilliz Cloud!"
      
        except Exception as e:
            
            warning_message = f"Zilliz connection failed: {e}. Falling back to temporary in-memory storage."
            vector_store = Chroma(embedding_function=embedding_model)
            return vector_store, "chroma", warning_message
    else:
        
        warning_message = "Zilliz credentials not found. Using temporary in-memory storage."
        vector_store = Chroma(embedding_function=embedding_model)
        return vector_store, "chroma", warning_message


def add_files_to_store(uploaded_files: List[Any], vector_store: Any, embedding_model: Any) -> str | None:
    all_chunks = []
    for uploaded_file in uploaded_files:
        text, err = get_text_from_uploaded_file(uploaded_file)
     
        if err:
            return f"Error reading {uploaded_file.name}: {err}"
      
        chunks, err = semantic_chunk(text, embedding_model)
      
        if err: 
            return f"Error chunking {uploaded_file.name}: {err}"
    
        all_chunks.extend(chunks)
   
    if not all_chunks:
        return "No text could be extracted from the uploaded files."
  
    try:
        prefixed_chunks = ["passage: " + chunk for chunk in all_chunks]
        vector_store.add_texts(texts=prefixed_chunks)
        return None
  
    except Exception as e:
        return f"An error occurred while adding documents to the vector store: {e}"