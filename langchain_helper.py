import re
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
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

    # ✅ Gemini LLM instead of Groq
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),   # or "gemini-1.5-pro"
        temperature=0.1,
        convert_system_message_to_human=True
    )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    sanitized_few_shots = sanitize_metadata(few_shots)
    to_vectorize = [" ".join(str(v) for v in example.values()) for example in few_shots]

    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=sanitized_few_shots)
    example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=2)

    mysql_prompt = """
        You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["query", "table_info", "top_k"],
    )

    chain = SQLDatabaseChain.from_llm(
        llm,
        db,
        verbose=True,
        prompt=few_shot_prompt,
    )
    return chain
