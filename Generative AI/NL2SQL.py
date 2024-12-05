######################################################################
# SQL Made Simple: How NL2SQL with Langchain is Reshaping Data Analysis
# Learn how NL2SQL with Langchain simplifies complex SQL queries, enabling # faster and more accessible data analysis through natural language commands

# Blog: https://medium.com/@amitvkulkarni/sql-made-simple-how-nl2sql-with-langchain-is-reshaping-data-analysis-9df763bdee33
######################################################################



######################################################################
# Imoprting the required libraries
######################################################################

import os


######################################################################
# Set up the database connection
######################################################################
db_user = "user"
db_password = "password"
db_host = "localhost"
db_name = "retail_sales_db"

from langchain_community.utilities.sql_database import SQLDatabase

######################################################################
# COnnecting to the database
######################################################################

db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
)

######################################################################
# Test the connection to the database
######################################################################
print(db.dialect)
print(db.get_usable_table_names())
print(db.table_info)


######################################################################
# Using the OpenAI API to generate SQL queries
######################################################################

from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
generate_query = create_sql_query_chain(llm, db)
query = generate_query.invoke(
    {"question": "How many transactions have more than 3 Quantity?`"}
)
print(query)


######################################################################
# Using prompt templates to generate the answer to the user question
######################################################################


from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

execute_query = QuerySQLDataBaseTool(db=db)
execute_query.invoke(query)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

rephrase_answer = answer_prompt | llm | StrOutputParser()

chain = (
    RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query") | execute_query
    )
    | rephrase_answer
)

chain.invoke({"question": "How many transactions have more than 3 Quantity?"})
