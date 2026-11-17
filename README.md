## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
AI-Powered PDF Question Answering Chatbot Students often spend a lot of time searching through lengthy PDF documents to find specific information. Reading entire documents to identify relevant answers can be time-consuming and difficult.

The proposed system aims to develop an AI-powered PDF Question Answering Chatbot using LangChain, OpenAI, and ChromaDB. Users can upload a PDF, ask questions in natural language, and receive relevant answers based on the document content.

Objective: To make information retrieval from PDF documents faster, easier, and more efficient.
### DESIGN STEPS:
1.Upload the PDF document using PyPDFLoader.

2.Extract and split the text into smaller chunks.

3.Convert text chunks into embeddings and store them in ChromaDB.

4.Retrieve relevant information based on the user's question.

5.Generate and display answers using the OpenAI language model.

### PROGRAM:
```
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load API Key
load_dotenv()

# Load PDF
loader = PyPDFLoader("AI_Notes.pdf")
documents = loader.load()

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create Vector Database
embeddings = OpenAIEmbeddings()
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# Create QA Chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=db.as_retriever()
)

# Ask Questions
while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    answer = qa.run(question)
    print("Bot:", answer)

```
### OUTPUT:
<img width="1600" height="1259" alt="image" src="https://github.com/user-attachments/assets/0c4568ec-3581-4407-87f3-8b2890bf4167" />


### RESULT:
The AI-powered PDF Question Answering Chatbot was successfully developed using LangChain, OpenAI, and ChromaDB. It extracts text from PDF documents, stores the text as embeddings, and retrieves relevant information based on user questions. The chatbot generates answers in natural language, making information retrieval faster, easier, and more efficient.
