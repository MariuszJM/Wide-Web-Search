# src/nodes.py

import os
import datetime
import json
import yaml
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage

# Platform-specific imports
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_community.tools import YouTubeSearchTool

# Utility imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings


def google_retrieve_urls(state):
    
    logger = state["logger"]
    search_queries = state["search_queries"]
    time_horizon = state["time_horizon"]
    sources_per_query = state["sources_per_query"]

    # Check API keys
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
        raise EnvironmentError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID environment variables.")

    search = GoogleSearchAPIWrapper()
    unique_urls = set()

    for search_query in search_queries:
        results = search.results(search_query, sources_per_query, search_params={'dateRestrict': f'd{time_horizon}', 'gl': 'EN'})
        urls = [item['link'] for item in results]
        unique_urls.update(urls)

    state["unique_urls"] = list(unique_urls)
    logger.info(f"Retrieved {len(unique_urls)} unique URLs from Google.")
    return state

def google_process_content(state):
    logger = state["logger"]
    unique_urls = state["unique_urls"]
    docs = []
    sources = {}
    for url in unique_urls:
        loader = WebBaseLoader(url)
        try:
            doc = loader.load()  
            docs.extend(doc)
            title = doc[0].metadata.get("title", url)
            content = doc[0].page_content
            sources[title] = {
                "url": url,
                "content": content
            }
        except Exception as e:
            logger.warning(f"Failed to load content from {url}: {e}")
    state["source_items"] = sources
    logger.info(f"Processed content from {len(docs)} Google documents.")
    return state

def youtube_retrieve_urls(state):
    logger = state["logger"]
    search_queries = state["search_queries"]
    sources_per_query = state["sources_per_query"]

    unique_urls = set()
    tool = YouTubeSearchTool()
    for search_query in search_queries:
        results = tool.run(search_query, 2 * sources_per_query)
        for item in results:
            unique_urls.add(item['url'])

    state["unique_urls"] = list(unique_urls)
    logger.info(f"Retrieved {len(unique_urls)} initial URLs from YouTube.")
    return state

def youtube_process_content(state):
    logger = state["logger"]
    unique_urls = state["unique_urls"]
    time_horizon = state["time_horizon"]
    sources_per_query = state["sources_per_query"]

    def youtube_get_content(urls):
        filtered_docs = []
        urls_within_time_horizon = []
        for url in urls:
            loader = YoutubeLoader.from_youtube_url(
                url, add_video_info=True
            )
            try:
                loaded_docs = loader.load()
                if loaded_docs:
                    # Assuming that 'publish_date' is in metadata of the document
                    doc = loaded_docs[0]
                    published_date_str = doc.metadata.get('publish_date')
                    if published_date_str:
                        try:
                            published_date = datetime.strptime(published_date_str, '%Y-%m-%dT%H:%M:%SZ')
                        except ValueError:
                            # Handle different date formats if necessary
                            published_date = datetime.strptime(published_date_str, '%Y-%m-%d')
                        if datetime.now() - published_date <= timedelta(days=time_horizon):
                            filtered_docs.extend(loaded_docs)
                            urls_within_time_horizon.extend(url)
                            if len(filtered_docs) >= sources_per_query:
                                break
            except Exception as e:
                logger.warning(f"Failed to load content from {url}: {e}")
        return filtered_docs, urls_within_time_horizon

    docs, urls_within_time_horizon = youtube_get_content(unique_urls)
    state["all_docs"] = docs
    state["unique_urls"] = urls_within_time_horizon
    logger.info(f"Processed content from {len(docs)} YouTube videos after filtering by time horizon.")
    return state


def create_embeddings(state):
    logger = state["logger"]
    all_docs = state["all_docs"]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(all_docs)
    # Add source information to each chunk
    for doc in doc_splits:
        doc.metadata['source'] = doc.metadata.get('source', 'unknown')

    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local", device="nvidia"),
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    state["retriever"] = retriever
    logger.info("Created embeddings for documents.")
    return state


def semantic_search_and_grading(state):
    logger = state["logger"]
    retriever = state["retriever"]
    specific_questions = state["specific_questions"]
    llm_json_mode = state["llm_json_mode"]
    relevant_chunks = {}
    for question in specific_questions:
        docs = retriever.invoke(question)
        for doc in docs:
            doc_txt = doc.page_content
            # Instructions for document grader
            doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""
            # Grader prompt
            doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

Carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' to indicate whether the document contains at least some information that is relevant to the question."""
            doc_grader_prompt_formatted = doc_grader_prompt.format(
                document=doc_txt, question=question
            )
            result = llm_json_mode.invoke(
                [SystemMessage(content=doc_grader_instructions)]
                + [HumanMessage(content=doc_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            if grade.lower() == "yes":
                source = doc.metadata.get('source', 'unknown')
                if source not in relevant_chunks:
                    relevant_chunks[source] = {}
                if question not in relevant_chunks[source]:
                    relevant_chunks[source][question] = []
                relevant_chunks[source][question].append(doc)
    state["relevant_chunks"] = relevant_chunks
    logger.info("Performed semantic search and grading.")
    return state


def generate_qa(state):
    logger = state["logger"]
    relevant_chunks = state["relevant_chunks"]
    llm = state["llm"]
    qa_results = {}
    for source, questions_docs in relevant_chunks.items():
        qa_results[source] = {}
        for question, docs in questions_docs.items():
            context = "\n\n".join(doc.page_content for doc in docs)
            # Prompt for generating answer
            rag_prompt = """You are an assistant for question-answering tasks.

Here is the context to use to answer the question:

{context}

Think carefully about the above context.

Now, review the user question:

{question}

Provide an answer to this question using only the above context.

Use three sentences maximum and keep the answer concise.

Answer:"""
            rag_prompt_formatted = rag_prompt.format(context=context, question=question)
            generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
            qa_results[source][question] = generation.content
    state["qa_results"] = qa_results
    logger.info("Generated Q&A for sources.")
    return state


def hallucination_check(state):
    logger = state["logger"]
    qa_results = state["qa_results"]
    llm_json_mode = state["llm_json_mode"]
    relevant_chunks = state["relevant_chunks"]
    valid_qa_results = {}
    for source, questions_answers in qa_results.items():
        valid_qa_results[source] = {}
        for question, answer in questions_answers.items():
            documents = "\n\n".join(doc.page_content for doc in relevant_chunks[source][question])
            # Instructions for hallucination grader
            hallucination_grader_instructions = """

You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS.

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of 'yes' means that the student's answer meets all of the criteria. This is the highest (best) score.

A score of 'no' means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""
            # Grader prompt
            hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}.

Return JSON with two keys: 'binary_score', which is 'yes' or 'no' to indicate whether the STUDENT ANSWER is grounded in the FACTS; and 'explanation', which contains an explanation of the score."""
            hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
                documents=documents,
                generation=answer
            )
            result = llm_json_mode.invoke(
                [SystemMessage(content=hallucination_grader_instructions)]
                + [HumanMessage(content=hallucination_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            if grade.lower() == "yes":
                valid_qa_results[source][question] = answer
    state["valid_qa_results"] = valid_qa_results
    logger.info("Performed hallucination check.")
    return state


def generate_summaries(state):
    logger = state["logger"]
    all_docs = state["all_docs"]
    llm = state["llm"]
    summaries = {}
    for doc in all_docs:
        source = doc.metadata.get('source', 'unknown')
        if source not in summaries:
            summaries[source] = {}
        content_length = len(doc.page_content)
        if content_length > 7500:
            # Generate detailed hierarchical summary
            summary_prompt = """Please provide a detailed hierarchical summary of the following content:

{content}

The summary should capture all the key points and structure of the content."""
            summary_prompt_formatted = summary_prompt.format(content=doc.page_content)
            summary = llm.invoke([HumanMessage(content=summary_prompt_formatted)])
            summaries[source]["detailed_summary"] = summary.content
        else:
            summaries[source]["content"] = doc.page_content

        # Generate one-sentence summary
        one_sentence_summary_prompt = """Please summarize the following content in one sentence:

{content}

Summary:"""
        one_sentence_summary_prompt_formatted = one_sentence_summary_prompt.format(content=doc.page_content)
        one_sentence_summary = llm.invoke([HumanMessage(content=one_sentence_summary_prompt_formatted)])
        summaries[source]["one_sentence_summary"] = one_sentence_summary.content

    state["summaries"] = summaries
    logger.info("Generated summaries for sources.")
    return state


def rank_sources(state):
    logger = state["logger"]
    valid_qa_results = state["valid_qa_results"]
    # Rank sources based on the number of correctly answered questions
    source_scores = {}
    for source, questions_answers in valid_qa_results.items():
        source_scores[source] = len(questions_answers)
    sorted_sources = sorted(source_scores.items(), key=lambda item: item[1], reverse=True)
    state["sorted_sources"] = sorted_sources
    logger.info("Ranked sources.")
    return state


def save_output(state):
    logger = state["logger"]
    sorted_sources = state["sorted_sources"]
    valid_qa_results = state["valid_qa_results"]
    summaries = state["summaries"]
    max_outputs = state["max_outputs"]
    llm_name = state["llm_name"]

    # Get current time for output folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"output_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    # Save configuration data
    config_data = {
        "search_queries": state["search_queries"],
        "specific_questions": state["specific_questions"],
        "platform": state["platform"],
        "time_horizon": state["time_horizon"],
        "max_outputs": state["max_outputs"],
        "llm_name": llm_name,
    }
    with open(os.path.join(output_folder, "config.yaml"), "w") as f:
        yaml.dump(config_data, f)

    best_sources = {}
    rest_sources = {}

    for idx, (source, score) in enumerate(sorted_sources):
        source_data = {
            "url": source,
            "one_sentence_summary": summaries.get(source, {}).get("one_sentence_summary", ""),
            "summary": summaries.get(source, {}).get("detailed_summary", ""),
            "content": summaries.get(source, {}).get("content", ""),
            "Q&A": valid_qa_results.get(source, {}),
        }
        if idx < max_outputs:
            best_sources[source] = source_data
        else:
            rest_sources[source] = source_data

    # Save best sources
    with open(os.path.join(output_folder, "best_sources.yaml"), "w") as f:
        yaml.dump(best_sources, f, allow_unicode=True)

    # Save rest of the sources
    with open(os.path.join(output_folder, "rest_sources.yaml"), "w") as f:
        yaml.dump(rest_sources, f, allow_unicode=True)

    # Log saving results
    state["output_folder"] = output_folder
    logger.info("Saved results to output folder.")
    return state
