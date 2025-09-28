import streamlit as st
import os
from StoreLogic import initialize_vector_store, add_files_to_store
from RetrievalLogic import handle_query 
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def get_embeddings_model():
    """Loads the embedding model once and caches it."""
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small", model_kwargs={'device': 'cpu'})


@st.cache_resource
def get_llm(api_key):
    """Connects to the LLM endpoint once and caches the connection."""
    return ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", huggingfacehub_api_token=api_key, temperature=0.5))




def main():
    st.set_page_config(page_title="AI Knowledge Base", page_icon=":Brain")
    st.header("Ask Your AI Knowledge Base ")


    hf_api_key = os.getenv("HF_API_KEY")
    zilliz_uri = os.getenv("ZILLIZ_CLOUD_URI")
    zilliz_api_key = os.getenv("ZILLIZ_CLOUD_API_KEY")
    zilliz_collection = os.getenv("ZILLIZ_COLLECTION_NAME")
    
    
    if not hf_api_key:
        st.error("Hugging Face API token not found in .env file."); return

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.store_type = "Initializing"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    embedding_model = get_embeddings_model()
    llm = get_llm(hf_api_key)


    if st.session_state.vector_store is None:
        zilliz_args = {"uri": zilliz_uri, "token": zilliz_api_key, "collection_name": zilliz_collection}
        with st.spinner("Connecting to Knowledge Base..."):
            vector_store, store_type, message = initialize_vector_store(embedding_model, zilliz_args)
            st.session_state.vector_store = vector_store
            st.session_state.store_type = store_type
            if message:
                st.toast(message)
                
    with st.sidebar:
        st.subheader("Add to Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload new documents", 
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        if st.button("Add to Database"):
            if uploaded_files:
                with st.spinner("Processing and adding documents..."):
                    err = add_files_to_store(uploaded_files, st.session_state.vector_store, embedding_model)
                    
                    if err:
                        st.error(err)
                    
                    else:
                        st.success(f"Successfully added {len(uploaded_files)} documents!")
           
            else:
                st.warning("Please upload at least one document.")


    if st.session_state.vector_store:
        st.caption(f"Connected to: {st.session_state.store_type.capitalize()} Database")
       
        for message in st.session_state.chat_history:
            
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if user_question := st.chat_input("Ask a question about your knowledge base..."):
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            
            with st.spinner("Thinking..."):
                bot_response, relevant_docs = handle_query(
                    llm,
                    st.session_state.vector_store,
                    user_question,
                    st.session_state.chat_history
                )
            
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)
                
                with st.expander("Show Sources"):
                    for i, doc in enumerate(relevant_docs):
                        st.info(f"Source {i+1}:\n\n{doc.page_content}")
    else:
        st.info("Knowledge base is initializing, please wait...")

if __name__ == '__main__':
    main()