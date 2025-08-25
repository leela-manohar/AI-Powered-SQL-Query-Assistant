
import streamlit as st
from langchain_helper import get_few_shot_db_chain

st.title("SQL LLM Assistant")

if "chain" not in st.session_state:
    st.session_state.chain = get_few_shot_db_chain()

question = st.text_input("Ask your database:")

if question:
    
    response = st.session_state.chain.run(question)

    
    if isinstance(response, str):
        if "Answer:" in response:
            final_answer = response.split("Answer:")[-1].strip()
        else:
            final_answer = response.strip()
    elif isinstance(response, dict) and "result" in response:
        final_answer = response["result"]
    else:
        final_answer = response

    st.write("### Answer")
    st.write(final_answer)

