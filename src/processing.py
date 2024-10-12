# src/processing.py

from src.utils import (
    create_retriever,
    is_relevant_chunk,
    generate_answer,
    check_hallucination,
    summarize_documents_map_reduce
)
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
            is_valid = check_hallucination(answer, relevant_chunks, llm_json)
            if is_valid.lower() == 'yes':
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
    return {'top_items': top_items, 'less_relevant_items': less_relevant_items}
