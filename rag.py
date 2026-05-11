from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders   import TextLoader, DirectoryLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
directory = "My Documents"

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

# Load documents from the specified directory
pdf_loader = PyPDFDirectoryLoader(directory)
pdf_docs = pdf_loader.load()

text_loader = DirectoryLoader(directory, glob = "**/*.txt", loader_cls=TextLoader)
text_docs = text_loader.load()

docs = pdf_docs + text_docs
print(f"Loaded {len(docs)} documents.")

# Chunking documents into smaller pieces for better retrieval performance
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")


# Creating embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)

user_query = input("\nPlease enter your query, or type exit: ")
retrieved_docs = vector_store.similarity_search(user_query, k=4)

#Source tracking for retrieved documents
for doc in retrieved_docs:
    print("Retrieved from:", doc.metadata.get("source"))
    if not retrieved_docs:
     print("No relevant documents found.")
     
context = "\n".join([doc.page_content for doc in retrieved_docs])


system_prompt = f"""
You are a helpful document analysis assistant.

Use the following document context to answer the user's question.

DOCUMENT CONTEXT:
{context}

If the answer is not in the document context, say that the document does not contain that information.
"""

messages = [
    SystemMessage(content=system_prompt),
]



messages.append(HumanMessage(content=user_query))

response = llm.invoke(messages)
print("\nAI RESPONSE:\n")
print(response)