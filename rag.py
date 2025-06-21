from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
import bs4

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="pdfs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


def process_pdf(file_path: str):
    global vector_store, embeddings
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    ids = vector_store.add_documents(documents=all_splits)


def process_website(url: str):
    global vector_store, embeddings

    loader = WebBaseLoader(web_paths=(url,), bs_kwargs={"features": "lxml"})

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    ids = vector_store.add_documents(documents=all_splits)


@tool(response_format="content")
def ingest_website(url: str):
    """
    Ingests a website and adds its content to the vector store.
    """
    process_website(url)
    return f"Website {url} has been ingested successfully."


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """
    Retrieves documents from the vector store based on the query.
    Returns the content of the retrieved documents and their metadata.
    """
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


llm = init_chat_model("openai:gpt-4o-mini")

tools = ToolNode([retrieve, ingest_website])

rag_agent = create_react_agent(
    model=llm,
    tools=[retrieve, ingest_website],
    prompt="Usted es un agente que recupera información de documentos y sitios web usando Retrieval-Augmented Generation (RAG)."
    "Si el usuario proporciona un URL, primero use la herramienta 'ingest_website' para agregar el contenido del sitio web a la base de datos de recuperación. "
    "Luego, use la herramienta 'retrieve' para responder a la consulta, citando los documentos o sitios web utilizados.",
    checkpointer=MemorySaver(),
    name="rag_agent",
)

__all__ = [
    "rag_agent",
    "retrieve",
    "ingest_website",
    "process_pdf",
    "process_website",
    "vector_store",
    "embeddings",
    "tools",
]
