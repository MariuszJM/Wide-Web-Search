from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


class ContentProcessor:
    """Class to handle content processing logic."""

    def __init__(self, llm_handler, llm_max_tokens):
        """Initialize with the given LLM and JSON LLM models."""
        self.llm_handler = llm_handler
        self.llm_max_tokens = llm_max_tokens

    def create_retriever(self, documents):
        """Create a retriever using document embeddings."""
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

    def is_relevant_chunk(self, chunk_text, question):
        """Determine if a document chunk is relevant to the given question."""
        instructions = """You are a grader assessing the relevance of a document to a user's question.
                        If the document contains keywords or semantic meaning related to the question, grade it as relevant."""
        prompt = f"""Document:\n\n{chunk_text}\n\nQuestion:\n\n{question}\n\nDoes the document contain information relevant to the question?
                    Return JSON with a single key 'binary_score' with value 'yes' or 'no'."""
        response = self.llm_handler.invoke_json(
            [SystemMessage(content=instructions), HumanMessage(content=prompt)]
        )

        return response.get("binary_score", "").lower() == "yes"

    def generate_answer(self, question, relevant_chunks):
        """Generate an answer based on relevant document chunks."""
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        prompt = f"""You are an assistant for answering questions.
                    Context:\n\n{context}\n\nQuestion:\n\n{question}
                    Provide a concise answer (maximum three sentences) based only on the above context."""
        response = self.llm_handler.invoke_text([HumanMessage(content=prompt)])
        return response.content.strip()

    def check_hallucination(self, answer, relevant_chunks):
        """Check if the generated answer is grounded in the document facts."""
        facts = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        instructions = """You are a teacher grading a student's answer based on provided facts.
                        Criteria:
                        1. The student's answer should be grounded in the facts.
                        2. The student's answer should not contain information outside the scope of the facts.
                        Return JSON with two keys: 'binary_score' ('yes' or 'no') indicating if the answer meets the criteria, and 'explanation' providing reasoning."""
        prompt = f"""Facts:\n\n{facts}\n\nStudent's Answer:\n\n{answer}\n\nIs the student's answer grounded in the facts?"""
        response = self.llm_handler.invoke_json(
            [SystemMessage(content=instructions), HumanMessage(content=prompt)]
        )
        return response.get("binary_score", "no")

    def summarize_documents_map_reduce(self, documents):
        """Summarize documents using a map-reduce approach."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.llm_max_tokens, chunk_overlap=self.llm_max_tokens // 10
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

        map_chain = map_prompt | self.llm_handler.llm | StrOutputParser()
        reduce_chain = reduce_prompt | self.llm_handler.llm | StrOutputParser()

        summaries = [map_chain.invoke(chunk.page_content) for chunk in doc_chunks]

        def calculate_total_tokens(summaries):
            return sum(
                self.llm_handler.llm.get_num_tokens(summary) for summary in summaries
            )

        def split_summaries_into_chunks(summaries, max_tokens):
            chunks, current_chunk, current_tokens = [], [], 0
            for summary in summaries:
                summary_tokens = self.llm_handler.llm.get_num_tokens(summary)
                if current_tokens + summary_tokens <= max_tokens:
                    current_chunk.append(summary)
                    current_tokens += summary_tokens
                else:
                    chunks.append(current_chunk)
                    current_chunk, current_tokens = [summary], summary_tokens
            if current_chunk:
                chunks.append(current_chunk)
            return chunks

        while calculate_total_tokens(summaries) > self.llm_max_tokens:
            chunks = split_summaries_into_chunks(summaries, self.llm_max_tokens)
            summaries = [reduce_chain.invoke("\n\n".join(chunk)) for chunk in chunks]

        final_summary = reduce_chain.invoke("\n\n".join(summaries))

        return final_summary

    def process_content(self, source_items, content_questions, max_top_sources):
        """Main method to process content using the LLM and return processed items."""
        processed_items = {}
        for title, data in source_items.items():
            documents = data["documents"]
            if not documents:
                continue    
            retriever = self.create_retriever(documents)
            qa_pairs = {}
            for question in content_questions:
                relevant_chunks = retriever.get_relevant_documents(question)
                relevant_chunks = [
                    chunk
                    for chunk in relevant_chunks
                    if self.is_relevant_chunk(chunk.page_content, question)
                ]
                if not relevant_chunks:
                    continue
                answer = self.generate_answer(question, relevant_chunks)
                if self.check_hallucination(answer, relevant_chunks).lower() == "yes":
                    qa_pairs[question] = answer
            if qa_pairs:
                summary = self.summarize_documents_map_reduce(documents)
                processed_items[title] = {
                    "url": data["url"],
                    "summary": summary,
                    "qa": qa_pairs,
                }
        ranked_items = sorted(
            processed_items.items(), key=lambda x: len(x[1]["qa"]), reverse=True
        )
        top_items = dict(ranked_items[:max_top_sources])
        less_relevant_items = dict(ranked_items[max_top_sources:])
        return {"top_items": top_items, "less_relevant_items": less_relevant_items}
