import streamlit as st
from uuid import uuid4
import os
import pprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY")
unique_id = uuid4().hex[0:8]
os.environ.update({
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_PROJECT": f"AI Research Agent - {unique_id}",
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
    "LANGCHAIN_API_KEY": langchain_api_key
})

# 1. Load the text document

file_path = "./RajeshRanjan_May_2025.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()
documents[0]
#pprint.pp(documents[0].metadata
for doc in documents:
    print(f"Preview: {doc.page_content[:100]}...")

# 2. Split the document into chunks


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)
print(f"Split into {len(split_docs)} chunks.")

# 3. Create embeddings and vector store
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embedding_model)
print("Vector store created.")
# 4. Create the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# 5. Retrieve relevant chunks for a query
query = "How long I worked with IBM? What was my role? What were my contributions?"
retrieved_docs = retriever.get_relevant_documents(query)
print(f"Retrieved {len(retrieved_docs)} relevant document(s):")
for i, doc in enumerate(retrieved_docs):
    print(f"\nChunk {i+1} preview:\n{doc.page_content[:200]}...")



# 6. Set up the prompt and LLM for answer generation


prompt = PromptTemplate.from_template(
    "Given the context below, answer the user's question:\n\n{context}\n\nQuestion: {question}"
)
llm = ChatOpenAI(model="gpt-4o")
output_parser = StrOutputParser()



# 7. Prepare the context and run the RAG chain
context = "\n\n".join(doc.page_content for doc in retrieved_docs)
chain_with_parser = prompt | llm | output_parser

response = chain_with_parser.invoke({
    "context": context,
    "question": query
})

print("\nFinal Answer:")
print(response)
