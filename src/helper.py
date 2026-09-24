
# =========================================================
# STEP 1: Define All Require Libraries
# =========================================================
import os
from dotenv import load_dotenv
from src.prompt import *
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



# =========================================================
# STEP 2: Groq authentication
# =========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# =========================================================
# STEP 3: Load PDF Document, Chunk Distribution
# =========================================================

def file_processing(file_path):
    """ 
    Method name: file_processing
    input : file path
    output: 
        - document_ques_gen: Return all document of 1st layer

        - document_answer_gen: Return Chunk of document of 2nd layer
    """

    # ---------------------------------------------------------
    # Step 3.1: Load data from PDF
    # ---------------------------------------------------------
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # # formatting the data
    question_gen = ''
    for page in data:
        question_gen += page.page_content

    # ---------------------------------------------------------
    # Step 3.2: 1st Layer of Chunk Distribution
    # ---------------------------------------------------------
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        # model_name = "openai/gpt-oss-20b",
        chunk_size=10000,
        chunk_overlap=200
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)


    # ---------------------------------------------------------
    # Step 3.3: 2nd Layer of Chunk Distribution
    # ---------------------------------------------------------
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]
    
    splitter_ans_gen = RecursiveCharacterTextSplitter(
        # model_name = "openai/gpt-oss-20b",
        chunk_size=1000,
        chunk_overlap=200
    )

    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen

# =========================================================
# STEP 4: llm pipeline 
# 
# =========================================================
def llm_pipeline(file_path):
    """ 
    input: file path
    output: 
        - return the all Questions answer: answer_generation_chain
        - Return the all Questions list: all_questions_lst
        
    """
    # ---------------------------------------------------------------------
    # Step 4.0: processing the provide file using file_processing function
    # ---------------------------------------------------------------------

    document_ques_gen, document_answer_gen = file_processing(file_path)

    # ---------------------------------------------------------
    # Step 4.1: Define the LLM to generate question
    # ---------------------------------------------------------
    llm_ques_gen_pipeline = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3
    )

    # -----------------------------------------------------------------------
    # Step 4.2: Define the Prompt 1 -> to create generate Questions
    # -----------------------------------------------------------------------
    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    
    # -----------------------------------------------------------------------
    # Step 4.3: Define the Prompt 2 -> to create Create Refine Questions
    # -----------------------------------------------------------------------
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    # -----------------------------------------------------------------------
    # Step 4.4: Define the Chian  -> to create Create Refine Questions
    # -----------------------------------------------------------------------
    Generic_Question_chain = (
        PROMPT_QUESTIONS
        | llm_ques_gen_pipeline
        | StrOutputParser()
    )

    ques_gen_chain = (
        {
            "existing_answer": Generic_Question_chain,
            "text": RunnablePassthrough()
        }
        | REFINE_PROMPT_QUESTIONS
        | StrOutputParser()

    )
    ques = ques_gen_chain.invoke(document_ques_gen)

    # -----------------------------------------------------------------------
    # Step 4.5: Define the Embedding Model
    # -----------------------------------------------------------------------
    embeddings =  HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------------------------------------------------
    # Step 4.5: Define the Vector Store (Knowledge Base)
    # -----------------------------------------------------------------------
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    # ---------------------------------------------------------
    # Step 4.6: Define the LLM for Answer
    # ---------------------------------------------------------
    llm_answer_gen = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.1
    )

    # Store the generated questions from all document chunks
    all_questions_lst = []
    for document in ques:
        text = document.page_content
        questions = llm_answer_gen.invoke(text)
        all_questions_lst.append(questions)

    # ---------------------------------------------------------
    # Step 4.7: Create a Retrieval
    # ---------------------------------------------------------
    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {
        "k": 4,
        }
    )

    # Helper to format retrieved documents into a single text block
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    # --------------------------------------------------------------------
    # Step 4.8: Define the Prompt for Generate Answer of the questions
    # --------------------------------------------------------------------
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])


    # ---------------------------------------------------------
    # Step 4.9: Create a Chain for Answer the questions
    # ---------------------------------------------------------
    answer_generation_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm_answer_gen
        | StrOutputParser()
    )

    return answer_generation_chain, all_questions_lst


