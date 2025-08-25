# import streamlit as st
# from langchain_helper import get_few_shot_db_chain

# st.title("SQL LLM Assistant")

# if "chain" not in st.session_state:
#     st.session_state.chain = get_few_shot_db_chain()

# question = st.text_input("Ask your database:")

# if question:
#     response = st.session_state.chain.run({"query": question})
#     st.write("Answer:", response)
import streamlit as st
from langchain_helper import get_few_shot_db_chain

st.title("SQL LLM Assistant")

if "chain" not in st.session_state:
    st.session_state.chain = get_few_shot_db_chain()

question = st.text_input("Ask your database:")

if question:
    # Get full response
    response = st.session_state.chain.run(question)

    # SQLDatabaseChain returns a dict if "return_intermediate_steps" is enabled
    # But in your case, it's just a string with all parts.
    # So let's extract only the "Answer:" part.
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
