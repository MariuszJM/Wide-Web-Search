platforms = [
    "No Fluff Jobs",
    "Pracuj.pl",
    "Solid.jobs",
    "Inhire.io",
    "Indeed",
    "LinkedIn",
    "Just Join IT"
]

search_queries_template = [
    "Machine Learning Engineer {platform}",
    "ML Ops Engineer {platform}",
    "Python Developer (AI/ML) {platform}",
    "AI Platform Engineer {platform}",
    "Data Scientist {platform}",
    "LLM AI Engineer {platform}",
]

content_questions = [
    "Is this a remote job offer?",
    "Is Python required?",
    "Is PyTorch needed?",
    "Is TensorFlow mentioned?",
    "Is Docker required?",
    "Are GitHub Actions needed?",
    "Are CI/CD needed?",
    "Is FastAPI required?",
    "Is Streamlit needed?",
    "Is Pandas required?",
    "Is Numpy needed?",
    "Is SQL required?",
    "Is LangChain mentioned?",
    "Is Matplotlib needed?",
    "Is Plotly required?",
    "Is Git required?",
    "Is English required?",
    "Is knowledge about RAG [Retrieval Augmented Generation] needed?",
    "Is knowledge about LLM [Large Language Model] needed?",
    "Is this a job offer page?",
    "Does this page contain a job offer?",
    "Is there a job offer on this page?",
    "Is this page presenting a job offer?",
    "Is a job offer provided on this page?",
    "Does this page have a job offer?",
]

for platform in platforms:
    filename = './input_tamplates/job_offers/' + platform.replace(".", "").replace(" ", "_") + ".yaml"
    with open(filename, "w") as f:
        f.write("SEARCH_QUERIES:\n")
        for query in search_queries_template:
            f.write(f'  - "{query.format(platform=platform)}"\n')
        f.write("\nCONTENT_QUESTIONS:\n")
        for question in content_questions:
            f.write(f'  - "{question}"\n')
        f.write("\nTIME_HORIZON_DAYS: 30\n")
        f.write("MAX_TOP_SOURCES: 10\n")
        f.write('PLATFORM: "google"\n')
        f.write("MAX_SOURCES_PER_SEARCH_QUERY: 10\n")
