# 🎤 Interview Question Creator — RAG-Based Question & Answer Generation

> Automatically generates interview questions and reference answers from any source document (PDF/book/resource) on a given topic, using a retrieval-augmented generation pipeline — exported as ready-to-use CSV.


---

## 📌 Problem

Preparing interview questions from learning resources is often a manual and time-consuming process. A learner typically needs to read PDFs, books, documentation, or other study materials, identify important concepts, understand the key points, and then manually create interview questions and answers from those materials.

**Real-world question:** *Can we automatically turn any source document into a reliable, grounded set of interview questions and answers?*.

This project aims to automate that process by building a Generative AI-powered Interview Question Creator that can analyze user-provided learning resources and automatically generate relevant interview questions with accurate answers based on the provided knowledge.

---


## 🎯 Objective

**Interview Question Creator** is a Generative AI application that uses **Retrieval-Augmented Generation (RAG)** to automatically generate interview questions and answers from user-provided learning resources. The system processes PDF documents using **LangChain** and **PyPDFLoader**, splits content into **token-based chunks**, generates semantic embeddings using **Hugging Face models**, and stores them in a **FAISS vector** index for similarity-based retrieval. Relevant document context is then passed to **GPT-OSS-20B** via Groq for grounded question and answer generation. Structured outputs are validated using schema-based parsing and exported as CSV, with **FastAPI** providing the backend API layer. The project is designed to evolve from notebook-based experimentation into a production-oriented GenAI pipeline with evaluation, validation, and deployment capabilities.


---

## 🧠 Solution Approaches

### 4.1 Approach 1 — Traditional Machine Learning / Deep Learning
• A traditional ML/DL approach could attempt to train a model specifically for question generation.

However, this approach introduces several challenges:

    ➔ Requires a large question-answer dataset.
    ➔ Requires significant training resources.
    ➔ Needs domain-specific training data.
    ➔ Requires continuous retraining when new learning resources are introduced.
    ➔ Difficult to directly ground generated questions in arbitrary user-provided PDFs.

Therefore, a pure ML/DL approach is not selected for the initial version.

### 4.2 Approach 2 — Generative AI + RAG (Selected)

• The selected approach is a Retrieval-Augmented Generation (RAG) pipeline.

Instead of training a new model, the system retrieves relevant information from the user's documents and provides that information as context to an LLM.

Core idea:
``` text 
User Resource
     ↓
Document Processing
     ↓
Knowledge Base
     ↓
Retrieve Relevant Information
     ↓
    LLM
     ↓
Generate Questions
     ↓
Generate Answers
     ↓
Validate & Structure
     ↓
CSV Output

```

## 🏗️ System Architecture

``` text

                         ┌──────────────────────────┐
                         │      USER RESOURCES      │
                         │                          │
                         │   PDF / Study Material   │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │     DOCUMENT LOADER      │
                         │                          │
                         │       PyPDFLoader        │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │      TEXT PROCESSING     │
                         │                          │
                         │     TokenTextSplitter    │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │     EMBEDDING MODEL      │
                         │                          │
                         │ HuggingFace Embeddings   │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │       VECTOR STORE       │
                         │                          │
                         │          FAISS           │
                         └─────────────┬────────────┘
                                       │
                                       │
═══════════════════════════════════════╪═══════════════════════════════
                                       │
                                USER QUERY
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │     QUERY EMBEDDING      │
                         │                          │
                         │     Embedding Model      │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │       RETRIEVER          │
                         │                          │
                         │    Similarity Search     │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │    RELEVANT CONTEXT      │
                         │                          │
                         │   Retrieved PDF Chunks   │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │       PROMPT 1           │
                         │                          │
                         │ Context + Topic + Rules  │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │          LLM             │
                         │                          │
                         │      GPT-OSS-20B         │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │   QUESTION GENERATION    │
                         │                          │
                         │ Questions + Difficulty   │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │       PROMPT 2           │
                         │                          │
                         │ Generated Questions      │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │          LLM             │
                         │                          │
                         │      Answer Generation   │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │   VALIDATION & PARSING   │
                         │                          │
                         │ Structured Output Schema │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │      CSV GENERATOR       │
                         │                          │
                         │ Question | Answer | ...  │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │      FINAL OUTPUT        │
                         │                          │
                         │       Interview.csv      │
                         └──────────────────────────┘

```

## 🧩 Tech Stack


---
| Component               | Tool                           |
| ----------------------- | ------------------------------------ |
| Project Type            | Generative AI / RAG                  |
| Programming Language    | Python                               |
| Orchestration Framework | LangChain                            |
| Document Loader         | PyPDFLoader                          |
| Text Splitter           | TokenTextSplitter                    |
| Embedding Model         | Hugging Face Sentence Transformers   |
| Candidate Embeddings    | all-MiniLM-L6-v2 / BGE-small-en-v1.5 |
| Vector Store            | FAISS                                |
| LLM                     | GPT-OSS-20B                          |
| LLM Inference Provider  | Groq                                 |
| Prompt Engineering      | LangChain Prompt Templates           |
| Structured Output       | Pydantic / LangChain                 |
| Backend API             | FastAPI                              |
| Data Processing         | Pandas                               |
| Output Format           | CSV                                  |
| Experimentation         | Jupyter Notebook                     |
| Version Control         | Git & GitHub                         |



## 🔧 Development Phases

The project is divided into two major phases:

## Phase 1 — Experimentation (Jupyter Notebook)

1. Objective
The objective of Phase 1 is to experimentally develop and validate the core components of the **Interview Question Creator** before moving to the production GenAI pipeline.
In this phase, the complete RAG-based workflow is implemented and tested inside a Jupyter Notebook. Each individual component is evaluated independently to ensure that document loading, text splitting, embedding generation, vector storage, retrieval, prompt engineering, question generation, answer generation, and structured output work correctly.

2. Phase 1 Workflow

    ```text
    PDF Document
        ↓
    Document Loading
        ↓
    Text Splitting
        ↓
    Text Embedding
        ↓
    FAISS Vector Store
        ↓
    Retriever
        ↓
    Relevant Context
        ↓
    Question Generation
        ↓
    Answer Generation
        ↓
    Structured Output
        ↓
    CSV Export
    ```

---


3. Load PDF Document
   - The project initially uses PDF documents as the primary knowledge source.

    - The PDF is successfully converted into LangChain `Document` objects containing:
        * Page content
        * Source information
        * Page metadata

---

4. Text Splitting
    - Large documents cannot efficiently be processed as a single text block. Therefore, the loaded document is divided into smaller chunks.


    - Why Token-Based Splitting?
        - The LLM processes information based on tokens. Therefore, token-based chunking provides better control over the amount of information passed to the model.

    - The original PDF has been transformed into smaller, manageable chunks that can later be embedded and stored in the vector database.

---

5. Analyze Chunk Distribution
    - Before creating embeddings, inspect the generated chunks to understand whether the selected chunk size is appropriate.

    - The chunk size should be large enough to preserve meaningful context but small enough to allow efficient retrieval and LLM processing.

---

6. Initialize Embedding Model
    - Convert each text chunk into a numerical vector representation.

    - The embedding model converts semantic information from text into vectors that can be compared using similarity search.

---

7. Generate Document Embeddings
    - Generate embeddings for all document chunks.

    - Instead of manually generating each embedding, the vector store integration will handle the embedding process.
    
    - Store the document embeddings in a vector index so that relevant information can be retrieved efficiently.


    - Architecture

        ```text
        PDF
         ↓
        Chunks
         ↓
        Embeddings
         ↓
        FAISS
        ```

    - The document knowledge base has now been converted into a searchable vector index.

---

8. Load Existing FAISS Vector Store
    - Test whether the previously created vector store can be loaded without processing the PDF again.

    - Persisting the vector store prevents unnecessary document processing and embedding generation every time the application starts.

---

9. Create Retriever
    - Convert the vector store into a retriever that can return the most relevant document chunks for a user query.

    - Retrieval Strategy

        ```text
        User Query
            ↓
        Query Embedding
            ↓
        FAISS Similarity Search
            ↓
        Top-K Relevant Chunks
        ```

---

10. Test Document Retrieval
    - Verify whether the retriever returns relevant information from the document.
   
    - The retrieved chunks should contain information relevant to the user's query.

    - If irrelevant chunks are retrieved, the retrieval configuration may need improvement.

---

11. Initialize LLM
    - Initialize the LLM that will generate interview questions and answers.

    - Why Use a Low Temperature?
        - A lower temperature is useful for this application because interview questions and answers should prioritize:
            * Accuracy
            * Consistency
            * Relevance
            * Reduced randomness

---

12. Create Question Generation Prompt
    - Design a prompt that instructs the LLM to generate interview questions based only on the retrieved document context.

    - Requirements:
        - Generate relevant technical interview questions.
        - Use the provided context as the primary knowledge source.
        - Do not introduce unsupported information.
        - Avoid duplicate questions.
        - Cover important concepts from the context.
        - Make the questions suitable for technical interviews.
        - Maintain the requested difficulty level.


---

13. Generate Interview Questions
    - Combine the retrieved context with the question-generation prompt and send it to the LLM.

    - The LLM generates questions based on the retrieved context rather than receiving the entire PDF.

---

14. Create Structured Question Schema
    - Instead of relying on free-form LLM responses, define a structured output schema.

    - Structured output makes the generated information easier to validate, process, and export.

---

15. Generate Structured Questions
    - Force the LLM to return the generated questions according to the predefined schema.

---

16.  Create Answer Generation Prompt

    - Generate accurate answers for the questions using the retrieved document context.

    - Requirements:
        - Answer the question clearly.
        - Use the provided context as the primary source.
        - Do not introduce unsupported claims.
        - Keep the answer technically accurate.
        - Provide enough explanation for interview preparation.


---

17. Generate Answers
    - Generate an answer for each generated interview question.

---

18. Validate Generated Results
    - Perform basic validation before exporting the generated content.

    - The system should verify:
        * Question is not empty.
        * Answer is not empty.
        * Topic is present.
        * Difficulty is valid.
        * Duplicate questions are removed.

---

19. Export Results to text
    - Convert the validated interview questions and answers into a txt file.
    
    - Final txt Structure
        ```text
        Topic,Difficulty,Question,Answer
        Machine Learning,Easy,What is overfitting?,...
        Machine Learning,Medium,What is Random Forest?,...
        Machine Learning,Hard,How does bagging reduce variance?,...
        ```

---


### 20. Phase 1 Experimentation Results

At the end of Phase 1, the following components should be successfully validated:

| Component            | Status |
| -------------------- | ------ |
| PDF Loading          | ☐      |
| Text Splitting       | ☐      |
| Embedding Generation | ☐      |
| FAISS Vector Store   | ☐      |
| Retriever            | ☐      |
| Retrieval Testing    | ☐      |
| LLM Integration      | ☐      |
| Question Generation  | ☐      |
| Answer Generation    | ☐      |
| Structured Output    | ☐      |
| Validation           | ☐      |
| CSV Export           | ☐      |

---

### 21. Experimentation Observations

During experimentation, record important observations for each component.

#### Document Loading

```text
Observation:
PDF successfully loaded and converted into LangChain Document objects.
```

#### Text Splitting

```text
Observation:
The document was divided into smaller token-based chunks to improve
retrieval and LLM context management.
```

#### Embedding

```text
Observation:
Document chunks were converted into semantic vector representations
using a Hugging Face embedding model.
```

#### Vector Store

```text
Observation:
FAISS successfully indexed the document embeddings and enabled
similarity-based retrieval.
```

#### Retrieval

```text
Observation:
The retriever returned the most relevant document chunks for the
provided technical query.
```

#### Generation

```text
Observation:
The LLM generated interview questions and answers using the retrieved
document context.
```

#### Structured Output

```text
Observation:
Structured output improved consistency and made the generated results
suitable for downstream processing.
```

#### CSV Export

```text
Observation:
The validated question-answer pairs were successfully exported into
a structured CSV file.
```

---



#### 22. Phase 1 → Phase 2 Transition

Once the complete pipeline has been validated in Jupyter Notebook, the implementation will move to **Phase 2 — Production GenAI Pipeline**.

#### Phase 1

```text
Jupyter Notebook
      ↓
Experiment
      ↓
Validate Components
      ↓
Find Best Configuration
```

The primary objective of Phase 1 is therefore not only to generate interview questions, but to **experimentally determine a reliable RAG pipeline that can later be transformed into a maintainable production GenAI application**.


## Phase 2 — Production Pipeline (FastAPI): 
After validating the individual components in Jupyter Notebook, the system will be converted into a production-oriented application.

### Production GenAI Pipeline
```text

                 ┌──────────────────────┐
                 │        USER          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     FastAPI API      │
                 └──────────┬───────────┘
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
      Upload PDF                      User Query
             │                             │
             ▼                             │
      Document Pipeline                    │
             │                             │
             ▼                             │
       Chunking + Embedding                │
             │                             │
             ▼                             │
          FAISS ◄──────────────────────────┘
             │
             ▼
        Retrieval Layer
             │
             ▼
       Prompt Pipeline
             │
             ▼
          LLM Layer
             │
       ┌─────┴─────┐
       ▼           ▼
  Questions      Answers
       │           │
       └─────┬─────┘
             ▼
       Validation
             │
             ▼
     Structured Output
             │
             ▼
       CSV Generator
             │
             ▼
        Final File


```

# Project Summary

**Interview Question Creator** is a Generative AI application that uses **Retrieval-Augmented Generation (RAG)** to automatically generate interview questions and answers from user-provided learning resources. The system processes PDF documents using **LangChain** and **PyPDFLoader**, splits content into **token-based chunks**, generates semantic embeddings using **Hugging Face models**, and stores them in a **FAISS vector** index for similarity-based retrieval. Relevant document context is then passed to **GPT-OSS-20B** via Groq for grounded question and answer generation. Structured outputs are validated using schema-based parsing and exported as CSV, with **FastAPI** providing the backend API layer. The project is designed to evolve from notebook-based experimentation into a production-oriented GenAI pipeline with evaluation, validation, and deployment capabilities.


