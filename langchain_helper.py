import re
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX

from few_shots import few_shots  # Your few-shot examples in a separate file

import os
from dotenv import load_dotenv

load_dotenv()


def clean_sql_text(sql_text: str) -> str:
    """
    Remove markdown code fences (```sql ... ```) from SQL query.
    Handles both ```sql and ``` cases.
    """
    if not sql_text:
        return sql_text
    sql_text = re.sub(r"```sql", "", sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r"```", "", sql_text)
    return sql_text.strip()


class CleanSQLDatabase(SQLDatabase):
    def run(self, command: str):
        """
        Override run() to clean SQL queries before executing them.
        """
        cleaned_command = clean_sql_text(command)
        return super().run(cleaned_command)


def sanitize_metadata(few_shots):
    sanitized = []
    for example in few_shots:
        clean_example = {}
        for k, v in example.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean_example[k] = v
            else:
                clean_example[k] = str(v)
        sanitized.append(clean_example)
    return sanitized


def get_few_shot_db_chain():
    db_user = "root"
    db_password = "1234"
    db_host = "localhost"
    db_name = "ig_clone"

    db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    db = CleanSQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)

    llm = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
    )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    sanitized_few_shots = sanitize_metadata(few_shots)
    to_vectorize = [" ".join(str(v) for v in example.values()) for example in few_shots]

    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=sanitized_few_shots)
    example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=2)

    mysql_prompt = """
        You are a helpful MySQL assistant. For each user question:
        1. Internally generate a correct SQL query.
        2. Execute the query on the database.
        3. Provide only the final answer in plain English—do NOT include the SQLQuery, SQLResult, or Question.

        If the question is ambiguous, ask for clarification.

        Examples:
        User: Who uploaded the most photos?
        Assistant: The user with the most photos is john_doe with 45 uploads.

        User: Which photo has the most likes?
        Assistant: The photo with ID 123 has the most likes, with 250 likes.

        Now answer the following question:
        """


    example_prompt = PromptTemplate(
    input_variables=["Question","Answer"],
    template="""
            Question: {Question}
            Answer: {Answer}""",
            )


    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix="Question: {query}\nAnswer:",
        input_variables=["query", "table_info", "top_k"],  # <-- expects "input"
    )

    chain = SQLDatabaseChain.from_llm(
        llm,
        db,
        verbose=True,
        prompt=few_shot_prompt,
    )
    return chain
