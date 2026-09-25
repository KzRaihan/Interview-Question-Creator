"""
Helper utilities for the Interview Question Creator.

This module provides the core RAG pipeline for:

1. Loading and processing PDF documents.
2. Creating hierarchical document chunks.
3. Generating contextual interview questions.
4. Creating a FAISS vector store.
5. Retrieving relevant context for each question.
6. Generating context-grounded answers.

The module uses:
    - LangChain
    - LangChain Community
    - LangChain Groq
    - Hugging Face Embeddings
    - FAISS
    - PyPDF
"""


# ============================================================
# STEP 1: Import Required Libraries
# ============================================================

import os
import re

from dotenv import load_dotenv

from src.prompt import (
    prompt_template,
    refine_template,
    system_prompt,
)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# STEP 2: Groq Authentication
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Please add GROQ_API_KEY to your .env file."
    )

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# ============================================================
# STEP 3: PDF File Processing
# ============================================================

def file_processing(file_path):
    """
    Load a PDF file and create two levels of document chunks.

    The first level creates relatively large chunks that are used
    for interview-question generation. The second level creates
    smaller chunks that are stored in the FAISS vector database
    for retrieval during answer generation.

    Parameters
    ----------
    file_path : str
        Path to the PDF file that will be processed.

    Returns
    -------
    tuple[list[Document], list[Document]]
        Returns two document collections:

        document_ques_gen:
            Large document chunks used for question generation.

        document_answer_gen:
            Smaller document chunks used for vector-store creation
            and retrieval during answer generation.

    Raises
    ------
    FileNotFoundError
        If the specified PDF file does not exist.
    """

    # ---------------------------------------------------------
    # Step 3.1: Validate the input file
    # ---------------------------------------------------------

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"PDF file was not found: {file_path}"
        )

    # ---------------------------------------------------------
    # Step 3.2: Load PDF document
    # ---------------------------------------------------------

    loader = PyPDFLoader(file_path)

    data = loader.load()

    if not data:
        raise ValueError(
            f"No content could be extracted from PDF: {file_path}"
        )

    # ---------------------------------------------------------
    # Step 3.3: Combine all PDF page content
    # ---------------------------------------------------------

    question_gen = "\n\n".join(
        page.page_content
        for page in data
        if page.page_content.strip()
    )

    if not question_gen.strip():
        raise ValueError(
            f"The PDF does not contain readable text: {file_path}"
        )

    # ---------------------------------------------------------
    # Step 3.4: First Layer of Chunking
    #
    # Large chunks are created for question generation.
    # ---------------------------------------------------------

    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=200,
    )

    chunks_ques_gen = splitter_ques_gen.split_text(
        question_gen
    )

    # Convert text chunks into LangChain Document objects.
    document_ques_gen = [
        Document(page_content=chunk)
        for chunk in chunks_ques_gen
    ]

    # ---------------------------------------------------------
    # Step 3.5: Second Layer of Chunking
    #
    # Smaller chunks are created for retrieval.
    # ---------------------------------------------------------

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen


# ============================================================
# STEP 4: Extract Individual Questions
# ============================================================

def extract_questions(question_text):
    """
    Extract individual questions from the LLM-generated response.

    The question-generation prompt instructs the LLM to return
    exactly 10 questions as a numbered list. This function converts
    that numbered response into a Python list containing individual
    question strings.

    Parameters
    ----------
    question_text : str
        Raw text returned by the question-generation LLM.

    Returns
    -------
    list[str]
        A list containing the individual generated questions.

    Raises
    ------
    ValueError
        If the LLM response does not contain any numbered questions.
    """

    # Store the extracted questions.
    questions = []

    # Temporary variable for handling multi-line questions.
    current_question = None

    # Process the LLM response line by line.
    for line in question_text.splitlines():

        line = line.strip()

        # Skip completely empty lines.
        if not line:
            continue

        # Check whether the line starts with a number.
        match = re.match(
            r"^\s*\d+[\.\)]\s*(.+)$",
            line
        )

        if match:

            # Save the previous question before starting
            # a new numbered question.
            if current_question is not None:
                questions.append(current_question.strip())

            # Start a new question.
            current_question = match.group(1).strip()

        elif current_question is not None:

            # If the current line is not numbered, treat it
            # as a continuation of the current question.
            current_question += " " + line

    # Add the final question.
    if current_question is not None:
        questions.append(current_question.strip())

    # Remove accidental empty values.
    questions = [
        question
        for question in questions
        if question
    ]

    if not questions:
        raise ValueError(
            "No numbered questions could be extracted from "
            "the LLM response."
        )

    return questions


# ============================================================
# STEP 5: Create Complete LLM Pipeline
# ============================================================

def llm_pipeline(file_path):
    """
    Build the complete interview-question generation and
    answer-generation pipeline for a PDF document.

    The pipeline performs the following operations:

    1. Loads and chunks the PDF.
    2. Generates initial interview questions.
    3. Refines the generated questions using additional context.
    4. Extracts individual questions from the LLM response.
    5. Creates Hugging Face embeddings.
    6. Builds a FAISS vector store.
    7. Creates a retriever for relevant context.
    8. Creates a context-grounded answer-generation chain.

    Parameters
    ----------
    file_path : str
        Path to the PDF file used as the knowledge source.

    Returns
    -------
    tuple
        Returns:

        answer_generation_chain:
            LangChain Runnable used to generate an answer for a
            selected interview question using retrieved context.

        all_questions_lst:
            List containing the individual interview questions
            generated from the PDF.

    Raises
    ------
    FileNotFoundError
        If the provided PDF file does not exist.

    ValueError
        If the PDF contains no readable content or the LLM fails
        to return numbered questions.
    """

    # ========================================================
    # Step 5.1: Process the PDF
    # ========================================================

    document_ques_gen, document_answer_gen = file_processing(
        file_path
    )

    # ========================================================
    # Step 5.2: Define LLM for Question Generation
    # ========================================================

    llm_ques_gen_pipeline = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
    )

    # ========================================================
    # Step 5.3: Create Initial Question Prompt
    # ========================================================

    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template,
        input_variables=["text"],
    )

    # ========================================================
    # Step 5.4: Create Question Refinement Prompt
    # ========================================================

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        template=refine_template,
        input_variables=[
            "existing_answer",
            "text",
        ],
    )

    # ========================================================
    # Step 5.5: Create Initial Question-Generation Chain
    # ========================================================

    Generic_Question_chain = (
        PROMPT_QUESTIONS
        | llm_ques_gen_pipeline
        | StrOutputParser()
    )

    # ========================================================
    # Step 5.6: Create Question Refinement Chain
    # ========================================================
    #
    # Flow:
    #
    # Input text
    #      |
    #      v
    # Initial Question Chain
    #      |
    #      v
    # Existing Questions
    #
    # Input text --------------------+
    #                                |
    #                                v
    #                         Refinement Prompt
    #                                |
    #                                v
    #                         Question LLM
    #                                |
    #                                v
    #                         Final Questions
    #
    # ========================================================

    ques_gen_chain = (
        {
            "existing_answer": Generic_Question_chain,
            "text": RunnablePassthrough(),
        }
        | REFINE_PROMPT_QUESTIONS
        | llm_ques_gen_pipeline
        | StrOutputParser()
    )

    # ========================================================
    # Step 5.7: Generate Questions for Each Document Chunk
    # ========================================================

    all_questions_lst = []

    for document in document_ques_gen:

        # Extract the actual text from the Document object.
        text = document.page_content

        # Generate and refine questions for this chunk.
        question_response = ques_gen_chain.invoke(text)

        # Convert the numbered LLM response into individual
        # question strings.
        questions = extract_questions(question_response)

        # Add the individual questions to the final list.
        all_questions_lst.extend(questions)

    # ========================================================
    # Step 5.8: Define Embedding Model
    # ========================================================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ========================================================
    # Step 5.9: Create FAISS Vector Store
    # ========================================================

    vector_store = FAISS.from_documents(
        document_answer_gen,
        embeddings,
    )

    # ========================================================
    # Step 5.10: Define LLM for Answer Generation
    # ========================================================

    llm_answer_gen = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.1,
    )

    # ========================================================
    # Step 5.11: Create Retriever
    # ========================================================

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,
        },
    )

    # ========================================================
    # Step 5.12: Format Retrieved Documents
    # ========================================================

    def format_docs(docs):
        """
        Combine retrieved documents into a single context string.

        Parameters
        ----------
        docs : list[Document]
            Documents returned by the vector-store retriever.

        Returns
        -------
        str
            Combined document content separated by blank lines.
        """

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    # ========================================================
    # Step 5.13: Create Answer Prompt
    # ========================================================

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # ========================================================
    # Step 5.14: Create RAG Answer-Generation Chain
    # ========================================================
    #
    # Question
    #    |
    #    +--------------------+
    #    |                    |
    #    v                    v
    # Retriever             Input
    #    |                    |
    #    v                    |
    # Relevant Context        |
    #    |                    |
    #    +---------+----------+
    #              |
    #              v
    #        Answer Prompt
    #              |
    #              v
    #          Answer LLM
    #              |
    #              v
    #        Final Answer
    #
    # ========================================================

    answer_generation_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm_answer_gen
        | StrOutputParser()
    )

    # ========================================================
    # Step 5.15: Return the Complete Pipeline
    # ========================================================

    return answer_generation_chain, all_questions_lst