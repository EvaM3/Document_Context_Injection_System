from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage
from pypdf import PdfReader

load_dotenv()


directory = Path("/Users/evam/Desktop/AI Agent/Document_Context_Injection_System/My Documents")

big_string = ""

user_query = input("Please enter your query: ")

def load_documents(directory="/Users/evam/Desktop/AI Agent/Document_Context_Injection_System/My Documents"):
    directory = Path(directory)
    documents = []

    for file_path in directory.rglob("*"):
        if file_path.suffix.lower() in {".txt", ".pdf", ".md"}:
            print("Loading:", file_path.name)

            if file_path.suffix.lower() == ".pdf":
                reader = PdfReader(file_path)
                content = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )
            else:
                content = file_path.read_text(encoding="utf-8")

            documents.append({
                "path": str(file_path),
                "content": content
            })

    return documents

documents = load_documents()
    
def create_context(document_list):
    context = ""
    for doc in document_list:
        context = context + f'{doc["path"]}:\n{doc["content"]}\n-------------------------\n'
    return context

context = create_context(documents)
print(context)

#big_string = create_context(documents)

print("\n===== DOCUMENT CONTEXT LOADED =====")
print(big_string[:3000])
print("===== END DOCUMENT CONTEXT =====\n")
print("Directory exists:", directory.exists())
print("Files found:", list(directory.rglob("*")))

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

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

while True:
    user_query = input("\nPlease enter your query, or type exit: ")

    if user_query.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    messages.append(HumanMessage(content=user_query))

    response = llm.invoke(messages)

    print("\nAI RESPONSE:\n")
    print(response)

    messages.append(SystemMessage(content=response))
