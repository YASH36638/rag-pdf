import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# Global vectorstore
vectorstore = None

# Load Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# Function to process PDF
def process_pdf(pdf_file):
    global vectorstore

    if pdf_file is None:
        return "Please upload a PDF file."

    # Load PDF
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    # Create vector database
    vectorstore = FAISS.from_documents(docs, embeddings)

    return f"PDF processed successfully.\nTotal Chunks Created: {len(docs)}"

# Function to answer questions
def ask_question(query):
    global vectorstore

    if vectorstore is None:
        return "Please upload and process a PDF first."

    # Similarity search
    matched_docs = vectorstore.similarity_search(query)

    # QA Chain
    chain = load_qa_chain(llm, chain_type="stuff")

    # Generate response
    response = chain.run(
        input_documents=matched_docs,
        question=query
    )

    return response

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# PDF Question Answering System")

    with gr.Row():
        pdf_input = gr.File(
            label="Upload PDF",
            file_types=[".pdf"]
        )

        upload_btn = gr.Button("Process PDF")

    upload_output = gr.Textbox(
        label="PDF Status"
    )

    upload_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=upload_output
    )

    gr.Markdown("## Ask Questions from PDF")

    question_input = gr.Textbox(
        label="Enter your question"
    )

    ask_btn = gr.Button("Ask")

    answer_output = gr.Textbox(
        label="Answer",
        lines=10
    )

    ask_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )

port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)