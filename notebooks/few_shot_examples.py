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
    SQLDatabase
    #SQLDatabaseChain,
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

import pyodbc


server = os.environ.get("RDS_SERVER_NAME")
database = os.environ.get("RDS_DB_NAME")
driver = os.environ.get("RSD_DRIVER_NAME")
openai_api_key=os.environ.get("OPENAI_API_KEY")

############### List of Examples
question_01 = "count of records in table medications ?"
question_02 = "What is the most used medications description?"

def create_examples(nlquery):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #llm = OpenAI(model_name="text-davinci-003", temperature=0, verbose=True, openai_api_key=openai_api_key)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=0, verbose=True, openai_api_key=openai_api_key)

    # define datasource uri
    # secret = get_secret(SECRET_NAME, REGION_NAME)
    # rds_uri = get_rds_uri(secret)


    #print("tototototototototototo")
    #print(driver)
    rds_uri = 'mssql+pyodbc:///?odbc_connect=' \
                'Driver='+ driver + \
                ';Server=' + server + \
                ';DATABASE=' + database + \
                ';integratedSecurity=true;trusted_Connection=yes;'


    db = SQLDatabase.from_uri(rds_uri)

    # load examples for few-shot prompting
    examples = load_samples()
    sql_db_chain = load_few_shot_chain(llm, db, examples)
    example: any
    try:
        output = sql_db_chain(nlquery)
        example = _parse_example(output)
    except Exception as exc:
        print("\n*** Query failed")
        result = {"query": nlquery, "intermediate_steps": exc.intermediate_steps}
        example = _parse_example(result)

    ######Examples Building, in reality you may want to write this out to a YAML file or database for manual fix-ups offline
    yaml_example = yaml.dump(example, allow_unicode=True)
    with open("few_shot_examples\sql_examples_mssql.yaml", 'a') as file:
        yaml.dump(example, file,  allow_unicode=True)
    print("\n" + yaml_example)


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
        return_intermediate_steps=True
    )

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

create_examples(question_02)

