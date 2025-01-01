# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:59:02 2024

@author: chand
"""

import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import streamlit as st
import data_loading_preprocessing



os.environ["OPENAI_API_KEY"] = "open-api-key"
db = FAISS.load_local("faiss_local_db", OpenAIEmbeddings(model='text-embedding-3-large'), allow_dangerous_deserialization=True)
print("FAISS index loaded from local storage.")
#############################################################
ret_em = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
# Query the retriever
ret_key = BM25Retriever.from_documents(data_loading_preprocessing.texts)
ret_key.k = 1
ensemble_retriever = EnsembleRetriever(retrievers= [ret_em, ret_key], weights=[0.8,0.2])

#############################################################

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
#############################################################
# Few-shot learning examples
examples = [
    {"input": "Color all alanine residues in blue.", "answer": "color #1/A:ala blue"},
    {"input": "Color the loop of the protein in yellow.", "answer": "color coil yellow."},
    {"input": "Color the sheet structure of the protein in blue and in transparency 50%.", "answer": "color strand blue transparency 80"},
    {"input": "Show the sequence of protein.", "answer": "ui tool show ```Show Sequence Viewer```"},
    {"input": "Color the ligand GV9 in dark blue.", "answer": "color :GV9 darkblue"},
]

# Few-shot Prompt Template
few_shot_template = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{answer}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=few_shot_template,
    examples=examples,
)

# Prompt for standalone question creation
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. "
    "Just formulate the query if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    ("human", "{input}")
])

# Prompt template for negotiation
negotiate_prompt = ChatPromptTemplate.from_messages([
    ("system","You are an expert in ChimeraX, specialized in generating its commands. Analyze queries to determine if they contain actionable instructions for tasks like visualization, analysis, or data manipulation. For actionable queries, provide precise, executable ChimeraX commands with clear steps."),
     # "You are a specialized assistant for ChimeraX tool. Provide clear and precise ChimeraX commands."),
    few_shot_prompt,
    ("human", "{input}"),
    ("human", "{context}")
])

# Memory for managing chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define retriever and chain
retriever = ensemble_retriever  # Your retriever (e.g., vectorstore retriever)

# History-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=contextualize_q_prompt
)

# Stuff Documents Chain
stuff_documents_chain = create_stuff_documents_chain(
    llm=llm, prompt=negotiate_prompt
)

# Full RAG Chain: Retrieval + QA
rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
)



