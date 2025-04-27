# === sql_trim_langgraph.py ===
# A trimmed LangGraph pipeline to only generate SQL query for evaluation

import os
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama

import re

# --- Setup database connection ---
# Get the database path relative to the project root
database_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                           "database", "bt5153_gp.db")
db = SQLDatabase.from_uri('sqlite:///' + database_path)

# --- Load Ollama LLM ---
llm = ChatOllama(model='qwen2.5:7b')

# --- Extract schema without comments ---
table_info = db.get_table_info()
table_info = table_info.split('/*')
for i in range(len(table_info)):
    if '*/' in table_info[i]:
        table_info[i] = table_info[i].split('*/')[1]
table_info = ''.join(table_info)

# --- LangGraph-compatible wrapper ---
def format_input(user_input):
    '''Format user input into a message for LangGraph.'''
    return {'messages': HumanMessage('User: ' + user_input)}

def write_query(state):
    '''Generate SQL query from question using LLM.'''
    question = state['messages'][-1].content.replace('User: ', '')
    prompt = (
        f'Generate a SQL query to answer the following question:\n\n'
        f'Question: {question}\n'
        f'Use the schema below:\n{table_info}\n\n'
        f'Strictly return only the SQL query as a code block, and do not use AS.'
    )
    response = llm.invoke(prompt)

    content = response.content
    if '```' in content:
        content = content.split('```')[1].strip('sql').strip()
    return {'query': content.strip()}

def remove_aliases(sql):
    # This safely removes "AS alias" without touching FROM or other clauses
    return re.sub(r'\s+AS\s+[a-zA-Z_][a-zA-Z0-9_]*', '', sql, flags=re.IGNORECASE)

# --- Graph setup (lightweight) ---
class State(TypedDict):
    messages: Annotated[list, lambda messages, new_message: messages + [new_message]]
    query: str

def create_graph():
    memory = MemorySaver()
    graph_builder = StateGraph(State)
    graph_builder.add_node('write_query', write_query)
    graph_builder.set_entry_point('write_query')
    graph_builder.set_finish_point('write_query')
    return graph_builder.compile()

graph = create_graph()

# --- Run LangGraph on your test set and save predictions ---
test_set = [
    {
        'question': 'How many patients incidents are there?',
    },
    {
        'question': 'How many inpatient cases had no treatment plan?',
    },
    {
        'question': "What is the average billing amount from each hospital's outpatient cases?",
    },
    {
        'question': 'What is the percentage of successful surgeries rount off to second decimal?',
    },
    {
        'question': "What is the average consultation time in minutes for each hospital's patient?",
    },
    {
        'question': 'How many inpatient cases that required surgery also had followups?',
    },
    {
        'question': 'What percentage of incidents involving patients over the age of 60 required surgeries?',
    },
    {
        'question': 'Which departments have the longest average surgery durations and also the highest average length of stay? Are these concentrated in specific hospitals?',
    },
    {
        'question': 'What is the average wait time from registration to consultation across hospitals, and how does it vary by department for outpatient cases?',
    },
    {
        'question': 'Among patients over 60 years old who had surgeries, which hospital and department combinations have the highest post-surgery readmission rates within 30 days?',
    }
]

# Run graph and collect predictions
pred_sql_list = []
for item in test_set:
    state_input = format_input(item['question'])
    result = graph.invoke(state_input)
    pred_sql = remove_aliases(result['query'])
    pred_sql_list.append(pred_sql)

# Save question + SQL on the same line
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "SPIDER")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "predicted_sql.txt")

with open(output_file, "w") as f:
    for sql in pred_sql_list:
        cleaned_sql = sql.replace('\n', ' ').strip()
        f.write(cleaned_sql + "\n")

print(f"✅ Predicted SQL queries saved to: {output_file}")
