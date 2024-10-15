from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import json
import logging
from config import LLM_MAX_TOKENS
from config import SPECIFIC_QUESTIONS, MAX_RESULTS

def process_content(source_items, llm, llm_json):
    processed_items = {}
    for title, data in source_items.items():
        documents = data['documents']
        retriever = create_retriever(documents)
        qa_pairs = {}
        for question in SPECIFIC_QUESTIONS:
            relevant_chunks = retriever.get_relevant_documents(question)
            relevant_chunks = [
                chunk for chunk in relevant_chunks if is_relevant_chunk(chunk.page_content, question, llm_json)
            ]
            if not relevant_chunks:
                continue
            answer = generate_answer(question, relevant_chunks, llm)
            if check_hallucination(answer, relevant_chunks, llm_json).lower() == 'yes':
                qa_pairs[question] = answer
        if qa_pairs:
            summary = summarize_documents_map_reduce(documents, llm)
            processed_items[title] = {
                'url': data['url'],
                'summary': summary,
                'qa': qa_pairs
            }
    ranked_items = sorted(processed_items.items(), key=lambda x: len(x[1]['qa']), reverse=True)
    top_items = dict(ranked_items[:MAX_RESULTS])
    less_relevant_items = dict(ranked_items[MAX_RESULTS:])
    return {
        'top_items': top_items,
        'less_relevant_items': less_relevant_items
    }

def create_retriever(documents):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    k = min(len(doc_chunks), 3)
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_chunks,
        embedding=NomicEmbeddings(
            model="nomic-embed-text-v1.5", inference_mode="local", device="nvidia"
        ),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever


def is_relevant_chunk(chunk_text, question, llm_json):
    instructions = """You are a grader assessing the relevance of a document to a user's question.

                    If the document contains keywords or semantic meaning related to the question, grade it as relevant."""
    prompt = f"""Document:\n\n{chunk_text}\n\nQuestion:\n\n{question}\n\nDoes the document contain information relevant to the question?

                Return JSON with a single key 'binary_score' with value 'yes' or 'no'."""
    response = llm_json.invoke(
        [SystemMessage(content=instructions), HumanMessage(content=prompt)]
    )
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        logging.warning(
            "LLM output is not a valid JSON. Please check your LLM model or the instructions."
        )
        return "no"
    return result.get("binary_score", "").lower() == "yes"


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
    response = llm_json.invoke(
        [SystemMessage(content=instructions), HumanMessage(content=prompt)]
    )
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        logging.warning(
            "LLM output is not a valid JSON. Please check your LLM model or the instructions."
        )
        return "no"
    return result.get("binary_score", "no")


def summarize_documents_map_reduce(documents, llm):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=LLM_MAX_TOKENS, chunk_overlap=0
    )
    doc_chunks = text_splitter.split_documents(documents)

    map_template = "You are an expert content summarizer. Combine your understanding of the following into a detailed nested bullet point summary:\n\n{context}"
    map_prompt = ChatPromptTemplate.from_messages([("human", map_template)])

    reduce_template = """
    The following is a set of summaries:
    {docs}
    Combine all of your understanding into a single, detailed nested bullet point summary with an overview at the beginning.
    """
    reduce_prompt = ChatPromptTemplate.from_messages([("human", reduce_template)])

    map_chain = map_prompt | llm | StrOutputParser()
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    summaries = [map_chain.invoke(chunk.page_content) for chunk in doc_chunks]

    def calculate_total_tokens(summaries):
        return sum(llm.get_num_tokens(summary) for summary in summaries)

    def split_summaries_into_chunks(summaries, max_tokens):
        chunks, current_chunk, current_tokens = [], [], 0
        for summary in summaries:
            summary_tokens = llm.get_num_tokens(summary)
            if current_tokens + summary_tokens <= max_tokens:
                current_chunk.append(summary)
                current_tokens += summary_tokens
            else:
                chunks.append(current_chunk)
                current_chunk, current_tokens = [summary], summary_tokens
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    while calculate_total_tokens(summaries) > LLM_MAX_TOKENS:
        chunks = split_summaries_into_chunks(summaries, LLM_MAX_TOKENS)
        summaries = [reduce_chain.invoke("\n\n".join(chunk)) for chunk in chunks]

    final_summary = reduce_chain.invoke("\n\n".join(summaries))

    return final_summary
