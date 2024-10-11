# src/utils.py

import os
import json
import yaml
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MAX_TOKENS

def create_retriever(documents):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_chunks,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local", device="nvidia"),
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 1})
    return retriever

def is_relevant_chunk(chunk_text, question, llm_json):
    instructions = """You are a grader assessing the relevance of a document to a user's question.

If the document contains keywords or semantic meaning related to the question, grade it as relevant."""
    prompt = f"""Document:\n\n{chunk_text}\n\nQuestion:\n\n{question}\n\nDoes the document contain information relevant to the question?

Return JSON with a single key 'binary_score' with value 'yes' or 'no'."""
    response = llm_json.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=prompt)
    ])
    result = json.loads(response.content)
    return result.get('binary_score', '').lower() == 'yes'

def generate_answer(question, relevant_chunks, llm):
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    prompt = f"""You are an assistant for answering questions.

Context:

{context}

Question:

{question}

Provide a concise answer (maximum three sentences) based only on the above context."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def check_hallucination(answer, relevant_chunks, llm_json):
    facts = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    instructions = """You are a teacher grading a student's answer based on provided facts.

Criteria:

1. The student's answer should be grounded in the facts.
2. The student's answer should not contain information outside the scope of the facts.

Return JSON with two keys: 'binary_score' ('yes' or 'no') indicating if the answer meets the criteria, and 'explanation' providing reasoning."""
    prompt = f"""Facts:

{facts}

Student's Answer:

{answer}

Is the student's answer grounded in the facts?"""
    response = llm_json.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=prompt)
    ])
    result = json.loads(response.content)
    return result.get('binary_score', 'no')

def summarize_documents(documents, llm):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=LLM_MAX_TOKENS, chunk_overlap=0
    )
    doc_chunks = text_splitter.split_documents(documents)
    map_prompt = ChatPromptTemplate.from_messages([
        ("human", "You are an expert summarizer. Summarize the following content into detailed bullet points:\n\n{context}")
    ])
    reduce_prompt = ChatPromptTemplate.from_messages([
        ("human", "Combine the following summaries into a single detailed summary:\n\n{summaries}")
    ])
    map_chain = map_prompt | llm | StrOutputParser()
    summaries = [map_chain.invoke(chunk.page_content) for chunk in doc_chunks]
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    combined_summary = reduce_chain.invoke("\n\n".join(summaries))
    return combined_summary

def create_output_directory(base_path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_results(processed_items, output_dir):
    top_items = processed_items.get('top_items', {})
    with open(os.path.join(output_dir, 'top_items.yaml'), 'w') as f:
        yaml.dump(top_items, f, default_flow_style=False)
    less_relevant_items = processed_items.get('less_relevant_items', {})
    with open(os.path.join(output_dir, 'less_relevant_items.yaml'), 'w') as f:
        yaml.dump(less_relevant_items, f, default_flow_style=False)
