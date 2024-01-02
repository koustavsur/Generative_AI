from typing import Any, Dict, List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()


# This method will use the retrieval QA to load the context from the vector and feed to LLM.
def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    # Provide the vector index created in local while ingesting
    new_vectorstore = FAISS.load_local("faiss_index_papi", embeddings)
    chat = ChatOpenAI(temperature=0, verbose=True)

    qa = ConversationalRetrievalChain.from_llm(llm=chat, chain_type="stuff", retriever=new_vectorstore.as_retriever(),
                                               return_source_documents=True)

    return qa({"question": query, "chat_history": chat_history})


# To run this program use streamlit. CMD -> streamlit run <absolute_path>/core.py
if __name__ == '__main__':
    st.header("Desktop Pay API Helper Bot")
    prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []

    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if prompt:
        with st.spinner("Generating response.."):
            response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
            result = response['answer']
            print(response)
            print(result)
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(result)
            st.session_state["chat_history"].append((prompt, result))

    if st.session_state["chat_answers_history"]:
        for generated_response, user_query in zip(
                st.session_state["chat_answers_history"],
                st.session_state["user_prompt_history"],
        ):
            message(user_query, is_user=True)
            message(generated_response)
