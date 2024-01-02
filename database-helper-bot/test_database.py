from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import tiktoken
import streamlit as st
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
from streamlit_chat import message

load_dotenv()


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# To run this program use streamlit. CMD -> streamlit run <absolute_path>/test_database.py
if __name__ == '__main__':
    # Provide postgresql+psycopg2://<username>:<password>@<hostname>:<port>/<DB-Name>
    db = SQLDatabase.from_uri('postgresql+psycopg2://psp_local:psp_local@localhost:5432/psp')
    print(db.get_usable_table_names())

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Provide argument verbose=True to see the agent(LLM) reasoning -> Action, Action input, Thoughts, Observations
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit
    )
    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []

    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []

    st.header("PSP Database Helper bot")
    prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
    print(prompt)

    if prompt:
        with st.spinner("Generating response.."):
            tokens = num_tokens_from_string(prompt, "gpt-3.5-turbo-16k")
            print(tokens)
            output = agent_executor.run(prompt)
            print(output)
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(str(output))

    if st.session_state["chat_answers_history"]:
        for generated_response, user_query in zip(
                st.session_state["chat_answers_history"],
                st.session_state["user_prompt_history"],
        ):
            message(user_query, is_user=True)
            message(generated_response)
