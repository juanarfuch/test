from langchain.prompts import PromptTemplate    

CONDENSE_PROMPT =QA_PROMPT=PromptTemplate("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:""")


template = QA_PROMPT = """As an AI, answer this transcript-based question. If uncertain or off-topic, admit it.

Context: {context}

Question: {question}

Answer in markdown. If unsure, say "Uncertain based on the transcript". Answer: """

QA_PROMPT=PromptTemplate(template=template, input_variables=["question", "context"])