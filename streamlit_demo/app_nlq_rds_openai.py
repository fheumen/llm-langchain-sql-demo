# Natural Language Query Demo of Amazon RDS for SQLServer using SageMaker FM Endpoint
# Author: Fabrice Heumeni
# Date: 2023-10-23
# Usage: streamlit run app.py --server.runOnSave true

import json
import logging
import os

import boto3
import yaml
from typing import Dict
from botocore.exceptions import ClientError
from langchain import (
    FewShotPromptTemplate,
    PromptTemplate,
    SQLDatabase,
    # SQLDatabaseChain,
)

from langchain_experimental.sql import SQLDatabaseChain

from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mssql_prompt
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.example_selector.semantic_similarity import (
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import Chroma
from streamlit_chat import message

import streamlit as st
import pyodbc


server = os.environ.get("RDS_SERVER_NAME")
database = os.environ.get("RDS_DB_NAME")
driver = os.environ.get("RSD_DRIVER_NAME")
openai_api_key = os.environ.get("OPENAI_API_KEY")


def main():
    st.set_page_config(page_title="Natural Language Query (NLQ) Demo")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # llm = OpenAI(model_name="text-davinci-003", temperature=0, verbose=True, openai_api_key=openai_api_key)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k-0613",
        temperature=0,
        verbose=True,
        openai_api_key=openai_api_key,
    )

    # define datasource uri
    # secret = get_secret(SECRET_NAME, REGION_NAME)
    # rds_uri = get_rds_uri(secret)

    # print("tototototototototototo")
    # print(driver)
    rds_uri = (
        "mssql+pyodbc:///?odbc_connect="
        "Driver="
        + driver
        + ";Server="
        + server
        + ";DATABASE="
        + database
        + ";integratedSecurity=true;trusted_Connection=yes;"
    )

    db = SQLDatabase.from_uri(rds_uri)

    # load examples for few-shot prompting
    examples = load_samples()

    sql_db_chain = load_few_shot_chain(llm, db, examples)

    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    if "query" not in st.session_state:
        st.session_state["query"] = []

    if "query_text" not in st.session_state:
        st.session_state["query_text"] = []

    # define streamlit colums
    col1, col2 = st.columns([4, 1], gap="large")

    # build the streamlit sidebar
    build_sidebar()

    # build the main app ui
    build_form(col1, col2)

    # get the users query
    get_text(col1)
    user_input = st.session_state["query"]

    if user_input:
        st.session_state.past.append(user_input)
        #example: any
        try:
            output = sql_db_chain.run(query=user_input)
            #example = _parse_example(output)
            st.session_state.generated.append(output)
            logging.info(st.session_state["query"])
            logging.info(st.session_state["generated"])
        except Exception as exc:
            st.session_state.generated.append(
                "I'm sorry, I was not able to answer your question."
            )
            logging.error(exc)

        ######Examples Building, in reality you may want to write this out to a YAML file or database for manual fix-ups offline
        # yaml_example = yaml.dump(example, allow_unicode=True)
        # with open("few_shot_examples\sql_examples_mssql.yaml", "a") as file:
        #     yaml.dump(example, file, allow_unicode=True)
        # print("\n" + yaml_example)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(
                st.session_state["generated"][i],
                key=str(i),
                is_user=False,
                avatar_style="icons",
                seed="459",
            )
            message(
                st.session_state["past"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="icons",
                seed="158",
            )


def get_secret(secret_name, region_name):
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    return json.loads(get_secret_value_response["SecretString"])


def get_rds_uri(secret):
    # SQLAlchemy 2.0 reference: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html
    # URI format: postgresql+psycopg2://user:pwd@hostname:port/dbname

    rds_username = secret["username"]
    rds_password = secret["password"]
    rds_endpoint = secret["host"]
    rds_port = secret["port"]
    rds_db_name = secret["dbname"]
    rds_db_name = "moma"
    return f"postgresql+psycopg2://{rds_username}:{rds_password}@{rds_endpoint}:{rds_port}/{rds_db_name}"


def load_samples():
    # Use the corrected examples for few-shot prompting examples
    sql_samples = None

    with open("few_shot_examples\sql_examples_mssql.yaml", "r") as stream:
        sql_samples = yaml.safe_load(stream)

    return sql_samples


def load_few_shot_chain(llm, db, examples):
    example_prompt = PromptTemplate(
        input_variables=["table_info", "input", "sql_cmd", "sql_result", "answer"],
        template="{table_info}\n\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {sql_result}\nAnswer: {answer}",
    )

    local_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        local_embeddings,
        Chroma,
        k=min(3, len(examples)),
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_mssql_prompt + "Here are some examples:",
        suffix=PROMPT_SUFFIX,
        input_variables=["table_info", "input", "top_k"],
    )

    return SQLDatabaseChain.from_llm(
        llm,
        db,
        prompt=few_shot_prompt,
        use_query_checker=True,
        verbose=True,
        return_intermediate_steps=False,
    )


def get_text(col1):
    with col1:
        input_text = st.text_input(
            "Ask a question:",
            "",
            key="query_text",
            placeholder="Your question here...",
            on_change=clear_text(),
        )
        logging.info(input_text)


def clear_text():
    st.session_state["query"] = st.session_state["query_text"]
    st.session_state["query_text"] = ""


def build_sidebar():
    with st.sidebar:
        st.title("Technologies")

        st.subheader("Natural Language Query (NLQ)")
        st.write(
            """
        [Natural language query (NLQ)](https://www.yellowfinbi.com/glossary/natural-language-query) enables analytics users to ask questions of their data. It parses for keywords and generates relevant answers sourced from related databases, with results typically delivered as a report, chart or textual explanation that attempt to answer the query, and provide depth of understanding.
        """
        )

        st.subheader("OMOP CDM Database")
        st.write(
            """
        [The Observational Medical Outcomes Partnership (OMOP) Common Data Model (CDM)](https://ohdsi.github.io/CommonDataModel/) contains tables allowing the systematic analysis of healthcare observational databases.
        """
        )

        st.subheader("Pycharm Professional Developers")
        st.write(
            """
        [Pycharm Pro ](https://www.jetbrains.com/pycharm/) is a fully integrated development environment (IDE) where you can perform all machine learning (ML) development steps, from preparing data to building, training, and deploying your ML models. 
        """
        )

        st.subheader("LangChain")
        st.write(
            """
        [LangChain](https://python.langchain.com/en/latest/index.html) is a framework for developing applications powered by language models.
        """
        )

        st.subheader("Chroma")
        st.write(
            """
        [Chroma](https://www.trychroma.com/) is the open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.
        """
        )

        st.subheader("Streamlit")
        st.write(
            """
        [Streamlit](https://streamlit.io/) is an open-source app framework for Machine Learning and Data Science teams. Streamlit turns data scripts into shareable web apps in minutes. All in pure Python. No front-end experience required.
        """
        )


def build_form(col1, col2):
    with col1:
        with st.container():
            st.title("Natural Language Query (NLQ) Demo")
            st.subheader("Ask questions of your data using natural language.")

        with st.container():
            with st.expander("Sample questions (copy and paste)"):
                st.text(
                    """
                How many artists are there in the collection?
                How many pieces of artwork are there in the collection?
                How many paintings are in the collection?
                How many artists are there whose nationality is French?
                How many artworks were created by Spanish artists?
                How many artist names start with the letter 'M'?
                Who is the most prolific artist in the collection? What is their nationality?
                What nationality of artists created the most artworks in the collection?
                What is the ratio of male to female artists? Return as ratio of n:1.
                How many artworks are by the artist, Claude Monet?
                What are the five oldest artworks in the collection? Return the title and date for each.
                For artist Frida Kahlo, return the title and medium of each artwork in a numbered list.
                Give me a recipe for chocolate cake.
                """
                )
    with col2:
        with st.container():
            st.button("clear chat", on_click=clear_session)


def clear_session():
    for key in st.session_state.keys():
        del st.session_state[key]


def _parse_example(result: Dict) -> Dict:
    sql_cmd_key = "sql_cmd"
    sql_result_key = "sql_result"
    table_info_key = "table_info"
    input_key = "input"
    final_answer_key = "answer"

    _example = {
        "input": result.get("query"),
    }

    steps = result.get("intermediate_steps")
    answer_key = sql_cmd_key  # the first one
    for step in steps:
        if isinstance(step, dict):
            if table_info_key not in _example:
                _example[table_info_key] = step.get(table_info_key)

            if input_key in step:
                if step[input_key].endswith("SQLQuery:"):
                    answer_key = sql_cmd_key  # this is the SQL generation input
                if step[input_key].endswith("Answer:"):
                    answer_key = final_answer_key  # this is the final answer input
            elif sql_cmd_key in step:
                _example[sql_cmd_key] = step[sql_cmd_key]
                answer_key = sql_result_key  # this is SQL execution input
        elif isinstance(step, str):
            _example[answer_key] = step
    return _example


if __name__ == "__main__":
    main()
