# Simple RAG Pipeline (PDF → FAISS → Groq)

This project demonstrates a clean, end-to-end implementation of a
Retrieval-Augmented Generation (RAG) system.

It shows how to take an unstructured document (a PDF), transform it into
a searchable knowledge base, and use a modern LLM to generate accurate,
context-aware answers. The goal is to provide a practical, minimal
example of how RAG systems work in real-world applications.

This system works best with PDF files that: - contain straightforward,
selectable text\
- are not scanned images\
- do not rely heavily on complex formatting

A sample file (`Trading-Volatility-Colin-Bennet.pdf`) is included and
was used to test this system. However, you can use **any PDF** by simply
changing the file path in the script.

------------------------------------------------------------------------

## Key concepts demonstrated

-   Text extraction and preprocessing\
-   Chunking with overlap for better context retention\
-   Embedding generation using sentence-transformers\
-   Vector search with FAISS\
-   Retrieval-Augmented Generation using an LLM\
-   Prompt design to reduce hallucinations

------------------------------------------------------------------------

## How it works

The pipeline has three steps:

1.  **Indexing**\
    Extract text from the PDF, split it into chunks, convert to
    embeddings, and store in FAISS.

2.  **Retrieval**\
    Search the most relevant chunks based on a user query.

3.  **Generation**\
    Send retrieved content to an LLM to generate a final answer.

------------------------------------------------------------------------

## Installation

Install the required dependencies:

``` bash
pip install pdfplumber numpy faiss-cpu sentence-transformers nltk groq
```

------------------------------------------------------------------------

## One-time setup (NLTK)

Before running the indexing step, download the tokenizer:

``` python
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
```

------------------------------------------------------------------------

## Step 1 --- Build the index

Run:

``` bash
python process_pdf.py
```

By default, the script uses:

``` python
pdf_path = "Trading-Volatility-Colin-Bennet.pdf"
```

You can replace this with any PDF file.

### Output

This step generates:

-   `faiss_index.bin` --- vector index\
-   `chunks.pkl` --- text chunks

------------------------------------------------------------------------

## Step 2 --- Test retrieval

Run:

``` bash
python query_vector_db.py
```

You can ask a question and see the most relevant chunks retrieved from
the document.

This step is useful to verify that your embeddings and search are
working correctly before using an LLM.

------------------------------------------------------------------------

## Step 3 --- Ask questions with an LLM

This step completes the RAG pipeline using Groq.

### API Key setup

Create a file:

    groq.txt

Add your API key inside:

    your_groq_api_key_here

### Run

``` bash
python query_groq.py
```

Ask a question, and the system will: - retrieve relevant chunks\
- send them to the model\
- generate a grounded answer

------------------------------------------------------------------------

## Notes

-   The system uses `all-MiniLM-L6-v2` for embeddings\
-   Retrieval uses FAISS with cosine similarity (via normalized
    embeddings)\
-   The LLM is instructed to only use retrieved context and say "I don't
    know" if the answer is not found

------------------------------------------------------------------------

## Project structure

    .
    ├── process_pdf.py
    ├── query_vector_db.py
    ├── query_groq.py
    ├── groq.txt
    ├── faiss_index.bin
    ├── chunks.pkl
    └── Trading-Volatility-Colin-Bennet.pdf

------------------------------------------------------------------------

## Summary

This project provides a practical, minimal implementation of a RAG
system that can:

-   process a PDF\
-   search its content efficiently\
-   generate answers grounded in the document

It is designed to be simple, extensible, and easy to adapt to other
datasets or use cases.
