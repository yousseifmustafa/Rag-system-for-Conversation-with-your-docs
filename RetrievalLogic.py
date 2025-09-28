from typing import List, Dict, Any, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


def retrieve_documents(user_question: str, vector_store: Any) -> List[Document]:
    """
    Searches the Vector DB and returns the most relevant document chunks.
    """
        
    retriever = vector_store.as_retriever(search_kwargs={'k': 4})
    return retriever.invoke(user_question)

def format_context(retrieved_docs: List[Document]) -> str:
    """Formats the retrieved documents into a single string for the prompt."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(f"--- Relevant Document {i+1} ---\n{doc.page_content}")
    return "\n---------------------------------\n".join(context_parts)

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Formats chat history into a readable string for the prompt."""
    if not chat_history:
        return "No past conversation history."
    
    history_str = ""
    for message in chat_history:
        history_str += f"{message['role'].capitalize()}: {message['content']}\n"
    return history_str

def create_final_prompt(user_question: str, retrieved_docs: List[Document], chat_history: List[Dict[str, str]]) -> str:
    """Prepares the final prompt for the LLM using a template."""
  
    context = format_context(retrieved_docs)
    formatted_history = format_chat_history(chat_history)

    template = """
You are a helpful AI assistant. Your goal is to answer the user's question based ONLY on the provided context.
If the answer is not found in the context, say "I could not find the answer in the provided document." and nothing more.

---
CHAT HISTORY:
{chat_history}
---
CONTEXT FROM DOCUMENT:
{context}
---
USER'S QUESTION:
{question}
---
YOUR ANSWER:
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )
    
    return prompt.format(
        chat_history=formatted_history,
        context=context,
        question=user_question
    )

def generate_answer(llm: Any, final_prompt: str) -> str:
    """Sends the final prompt to the LLM and returns the content of the response."""
    response = llm.invoke(final_prompt)
    return response.content


def handle_query(llm: Any, vector_store: Any, user_question: str, chat_history: List[Dict[str, str]]) -> Tuple[str, List[Document]]:
    """
    Orchestrates the entire RAG pipeline: retrieve, format prompt, and generate answer.
    Returns the final answer and the retrieved documents for citation.
    """

    relevant_docs = retrieve_documents(user_question, vector_store)
    
    final_prompt = create_final_prompt(user_question, relevant_docs, chat_history)
    
    bot_response = generate_answer(llm, final_prompt)
    
    return bot_response, relevant_docs