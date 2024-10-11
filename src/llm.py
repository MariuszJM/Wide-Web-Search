from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

def get_llm(llm_name):
    if llm_name == "ollama":
        local_llm = "llama3.2:latest"
        llm = ChatOllama(model=local_llm, temperature=0)
        return llm
    elif llm_name == "groq":
        local_llm = "llama3-8b-8192"
        llm = ChatGroq(model=local_llm, temperature=0.0)
        return llm
    else:
        raise ValueError(f"Unknown LLM name: {llm_name}")

def get_llm_json_mode(llm_name):
    if llm_name == "ollama":
        local_llm = "llama3.2:latest"
        llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")
        return llm_json_mode
    elif llm_name == "groq":
        local_llm = "llama3-8b-8192"
        llm_json_mode = ChatGroq(model=local_llm, temperature=0.0).with_structured_output(
            method="json_mode"
        )
        return llm_json_mode
    else:
        raise ValueError(f"Unknown LLM name: {llm_name}")
