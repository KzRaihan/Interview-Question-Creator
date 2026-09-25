"""
FastAPI application for the Interview Question Creator.

This application provides the web interface and API endpoints
required to:

1. Display the Interview Question Creator frontend.
2. Upload PDF learning materials.
3. Validate uploaded PDF files.
4. Generate interview questions and answers using the RAG pipeline.
5. Export generated questions and answers to a CSV file.

Application architecture:

Browser
   |
   | POST /upload
   v
PDF Upload
   |
   v
PDF Validation
   |
   | POST /analyze
   v
RAG Pipeline
   |
   +--> Question Generation
   |
   +--> FAISS Retrieval
   |
   +--> Answer Generation
   |
   v
QA.csv
   |
   v
Browser Download
"""

# ============================================================
# STEP 1: Import Required Libraries
# ============================================================

import csv
import os
from pathlib import Path

import uvicorn
import aiofiles

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pypdf import PdfReader

from src.helper import llm_pipeline

# ============================================================
# STEP 2: Application Configuration
# ============================================================

# ------------------------------------------------------------
# Application directories
# ------------------------------------------------------------

STATIC_DIR = Path("static")
DOCUMENTS_DIR = STATIC_DIR / "docs"
OUTPUT_DIR = STATIC_DIR / "output"
TEMPLATES_DIR = Path("templates")

# ------------------------------------------------------------
# Application constraints
# ------------------------------------------------------------

MAX_PDF_PAGES = 5
ALLOWED_CONTENT_TYPE = "application/pdf"

# ------------------------------------------------------------
# Create required directories
# ------------------------------------------------------------

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 3: Initialize FastAPI Application
# ============================================================

app = FastAPI(
    title="Interview Question Creator",
    description=(
        "AI-powered application for generating interview "
        "questions and answers from PDF learning materials."
    ),
    version="1.0.0",
)

# ============================================================
# STEP 4: Configure Static Files and Templates
# ============================================================

# Serve uploaded PDFs and generated CSV files through /static.
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

# Configure Jinja2 templates.
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ============================================================
# STEP 5: Default Route
# ============================================================
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )

# @app.get("/")
# async def index(request: Request):
#     """
#     Render the Interview Question Creator homepage.

#     Parameters
#     ----------
#     request : Request
#         FastAPI request object used by Jinja2 to render the template.

#     Returns
#     -------
#     TemplateResponse
#         Rendered `index.html` page.
#     """

#     return templates.TemplateResponse(
#         "index.html",
#         {
#             "request": request
#         },
#     )


# ============================================================
# STEP 6: Validate PDF File
# ============================================================

def validate_pdf(file_path: Path):
    """
    Validate an uploaded PDF file.

    The function checks whether the file can be opened as a PDF
    and whether its number of pages is within the application's
    maximum allowed limit.

    Parameters
    ----------
    file_path : Path
        Path to the uploaded PDF file.

    Returns
    -------
    int
        Number of pages contained in the PDF.

    Raises
    ------
    HTTPException
        Raised when the file is not a valid PDF or when the
        maximum page limit is exceeded.
    """

    try:
        # Open the PDF using pypdf.
        reader = PdfReader(str(file_path))

        # Count the PDF pages.
        page_count = len(reader.pages)

    except Exception as exc:
        # Remove invalid uploaded file.
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        ) from exc

    # --------------------------------------------------------
    # Check maximum page limit
    # --------------------------------------------------------

    if page_count > MAX_PDF_PAGES:
        # Remove the uploaded file because it is not accepted.
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=400,
            detail=(
                f"The PDF contains {page_count} pages. "
                f"The maximum allowed number of pages is "
                f"{MAX_PDF_PAGES}."
            ),
        )

    return page_count


# ============================================================
# STEP 7: Upload PDF
# ============================================================

@app.post("/upload")
async def upload_pdf(
    pdf_file: UploadFile = File(...),
):
    """
    Upload and validate a PDF learning resource.

    The endpoint:

    1. Checks that a file was provided.
    2. Checks that the uploaded file is a PDF.
    3. Saves the PDF into `static/docs/`.
    4. Validates the PDF page count.
    5. Returns the browser-accessible PDF path.

    Parameters
    ----------
    pdf_file : UploadFile
        PDF file uploaded from the frontend.

    Returns
    -------
    JSONResponse
        JSON response containing the uploaded PDF path.

    Raises
    ------
    HTTPException
        Raised when the uploaded file is invalid or cannot be saved.
    """

    # --------------------------------------------------------
    # Step 7.1: Validate filename
    # --------------------------------------------------------

    if not pdf_file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was selected.",
        )

    # --------------------------------------------------------
    # Step 7.2: Validate file extension
    # --------------------------------------------------------

    original_filename = Path(pdf_file.filename).name

    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    # --------------------------------------------------------
    # Step 7.3: Validate content type
    # --------------------------------------------------------

    if (
        pdf_file.content_type
        and pdf_file.content_type != ALLOWED_CONTENT_TYPE
    ):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must be a PDF.",
        )

    # --------------------------------------------------------
    # Step 7.4: Create safe output filename
    # --------------------------------------------------------

    safe_filename = Path(original_filename).name
    pdf_path = DOCUMENTS_DIR / safe_filename

    # --------------------------------------------------------
    # Step 7.5: Save uploaded PDF
    # --------------------------------------------------------

    try:
        async with aiofiles.open(pdf_path, "wb") as output_file:
            while True:
                chunk = await pdf_file.read(1024 * 1024)

                if not chunk:
                    break

                await output_file.write(chunk)

    except Exception as exc:
        if pdf_path.exists():
            pdf_path.unlink()

        raise HTTPException(
            status_code=500,
            detail="Failed to save the uploaded PDF.",
        ) from exc

    finally:
        await pdf_file.close()

    # --------------------------------------------------------
    # Step 7.6: Validate PDF
    # --------------------------------------------------------

    page_count = validate_pdf(pdf_path)

    # --------------------------------------------------------
    # Step 7.7: Return browser-accessible path
    # --------------------------------------------------------

    browser_pdf_path = f"/static/docs/{safe_filename}"

    return JSONResponse(
        status_code=200,
        content={
            "msg": "success",
            "pdf_filename": browser_pdf_path,
            "page_count": page_count,
        },
    )


# ============================================================
# STEP 8: Generate CSV File
# ============================================================

def get_csv(file_path: str):
    """
    Generate interview questions and answers and save them to CSV.

    The function executes the RAG pipeline created by
    `llm_pipeline()`. It receives the generated questions,
    retrieves relevant context for each question, generates
    an answer, and stores the question-answer pairs in a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the uploaded PDF file.

    Returns
    -------
    str
        Browser-accessible path to the generated CSV file.

    Raises
    ------
    ValueError
        If the RAG pipeline does not generate any questions.

    FileNotFoundError
        If the provided PDF file does not exist.

    Exception
        If question or answer generation fails.
    """

    # --------------------------------------------------------
    # Step 8.1: Validate input file
    # --------------------------------------------------------

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # --------------------------------------------------------
    # Step 8.2: Build the RAG pipeline
    # --------------------------------------------------------

    answer_generation_chain, ques_list = llm_pipeline(file_path)

    # --------------------------------------------------------
    # Step 8.3: Validate generated questions
    # --------------------------------------------------------

    if not ques_list:
        raise ValueError(
            "No interview questions were generated from the PDF."
        )

    # --------------------------------------------------------
    # Step 8.4: Define CSV output file
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / "QA.csv"

    # --------------------------------------------------------
    # Step 8.5: Write questions and answers to CSV
    # --------------------------------------------------------

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write CSV header.
        csv_writer.writerow(["Question", "Answer"])

        # ----------------------------------------------------
        # Generate an answer for every question
        # ----------------------------------------------------

        for question in ques_list:
            # Remove unnecessary whitespace.
            question = question.strip()

            if not question:
                continue

            # Generate answer using the RAG chain.
            #
            # IMPORTANT:
            # The updated helper.py returns a LangChain Runnable.
            # Therefore `.invoke()` must be used instead of `.run()`.
            answer = answer_generation_chain.invoke(question)

            # ------------------------------------------------
            # Save question-answer pair
            # ------------------------------------------------

            csv_writer.writerow([question, answer])

    # --------------------------------------------------------
    # Step 8.6: Return browser-accessible CSV path
    # --------------------------------------------------------

    return "/static/output/QA.csv"


# ============================================================
# STEP 9: Analyze PDF and Generate Q&A
# ============================================================

@app.post("/analyze")
async def analyze_pdf(
    pdf_filename: str = Form(...),
):
    """
    Analyze an uploaded PDF and generate interview questions
    and answers.

    The endpoint receives the PDF path returned by `/upload`,
    executes the RAG pipeline, generates question-answer pairs,
    and creates the final CSV file.

    Parameters
    ----------
    pdf_filename : str
        Browser-relative path of the uploaded PDF.

    Returns
    -------
    JSONResponse
        JSON response containing the generated CSV file path.

    Raises
    ------
    HTTPException
        Raised when the PDF cannot be found or processing fails.
    """

    # --------------------------------------------------------
    # Step 9.1: Validate input
    # --------------------------------------------------------

    if not pdf_filename:
        raise HTTPException(
            status_code=400,
            detail="PDF filename is required.",
        )

    # --------------------------------------------------------
    # Step 9.2: Convert frontend path to local path
    # --------------------------------------------------------

    filename = Path(pdf_filename).name
    local_pdf_path = DOCUMENTS_DIR / filename

    # --------------------------------------------------------
    # Step 9.3: Verify PDF exists
    # --------------------------------------------------------

    if not local_pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded PDF file was not found.",
        )

    # --------------------------------------------------------
    # Step 9.4: Generate questions and answers
    # --------------------------------------------------------

    try:
        output_file = get_csv(str(local_pdf_path))

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Log the actual error in the server console.
        print(f"Error while processing PDF: {exc}")

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while generating "
                "the interview questions and answers."
            ),
        ) from exc

    # --------------------------------------------------------
    # Step 9.5: Return generated file
    # --------------------------------------------------------

    return JSONResponse(
        status_code=200,
        content={
            "output_file": output_file,
        },
    )


# ============================================================
# STEP 10: Application Entry Point
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )