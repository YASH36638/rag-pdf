import os
from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide the main tkinter window
Tk().withdraw()

# Open file dialog to select PDF
pdf_path = askopenfilename(
    title="Select a PDF File",
    filetypes=[("PDF Files", "*.pdf")]
)

# Print the selected PDF path
if pdf_path:
    print("Selected PDF Path:")
    print(pdf_path)
else:
    print("No PDF file selected.")


# Step 1: Load pdf
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print("PDF successfully loaded....")

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

docs = text_splitter.split_documents(documents)
print('Chunks Created', len(docs))


# Step 3: Create Embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
print('Embedding model loaded')

vectorstore = FAISS.from_documents(docs, embeddings)
print('Vector Database Created')


# Step 4: Load Gemini Model
llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    temperature = 0.3,
    google_api_key = os.getenv("GOOGLE_API_KEY")
)
print("LLM loaded")


# STep 5: Ask Question
query = input('Ask Question :')

matched_docs = vectorstore.similarity_search(query)

chain = load_qa_chain(llm, chain_type='stuff')

response = chain.run(input_documents=matched_docs, question=query)

print('Response :')
print(response)