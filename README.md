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
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Load PDF
loader = PyPDFLoader("Generative_AI_Study_Guide_10_Pages.pdf")
documents = loader.load()

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings and store in ChromaDB
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="docs/chroma"
)

# 4. Retrieve relevant information
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3})
)

# 5. Generate and display answer
question = "What is Generative AI?"
result = qa_chain.invoke({"query": question})
print("Question:", question)
print("Answer:", result["result"])
```
### OUTPUT:
<img width="1930" height="815" alt="image" src="https://github.com/user-attachments/assets/5f7d5217-877d-4e83-9c15-d950bda1f903" />

### RESULT:
The AI-powered PDF Question Answering Chatbot was successfully developed using LangChain, OpenAI, and ChromaDB. It extracts text from PDF documents, stores the text as embeddings, and retrieves relevant information based on user questions. The chatbot generates answers in natural language, making information retrieval faster, easier, and more efficient.
