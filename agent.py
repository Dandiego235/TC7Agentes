from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_tavily import TavilySearch
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from rag import rag_agent
from IPython.display import Image, display
import os

load_dotenv()

# Código realizado con base en los tutoriales de LangGraph y LangChain:
# ‌Build an Agent. (n.d.). LangChain. Recuperado el 23 de junio, 2025, de https://python.langchain.com/docs/tutorials/agents/
tools_list = load_tools(["arxiv"])
tavily_search_tool = TavilySearch(max_results=3)
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

math_assistent = create_react_agent(
    model="openai:gpt-4o",
    tools=[],
    prompt="Usted es un asistente experto en matemáticas."
    "Responda a las preguntas de los usuarios de manera clara y concisa."
    "Explique el paso a paso de cómo resolver los problemas matemáticos."
    "Utilice el lenguaje adecuado para un estudiante de secundaria.",
    name="math_assistant",
    checkpointer=MemorySaver(),
)

history_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[tavily_search_tool, wikipedia_tool],
    prompt="Usted es un asistente experto en historia."
    "Responda a las preguntas de los usuarios de manera clara y concisa."
    "Explique el contexto histórico y los eventos relevantes."
    "Utilice la búsqueda en línea para proporcionar información actualizada y precisa."
    "Utilice el lenguaje adecuado para un estudiante de secundaria.",
    name="history_assistant",
    checkpointer=MemorySaver(),
)

science_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[tavily_search_tool, wikipedia_tool] + tools_list,
    prompt="Usted es un asistente experto en ciencias."
    "Responda a las preguntas de los usuarios de manera clara y concisa."
    "Explique el concepto científico y los principios relacionados."
    "Utilice el lenguaje adecuado para un estudiante de secundaria.",
    name="science_assistant",
    checkpointer=MemorySaver(),
)

spanish_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[],
    prompt="Usted es un asistente experto en español."
    "Responda a las preguntas de los usuarios de manera clara y concisa."
    "Revise la redacción, la gramática y la ortografía de los textos en español, corrigiendo errores y mejorando la claridad."
    "Responda preguntas sobre literatura en español."
    "Utilice el lenguaje adecuado para un estudiante de secundaria.",
    name="spanish_assistant",
    checkpointer=MemorySaver(),
)

recommender_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[tavily_search_tool, wikipedia_tool] + tools_list,
    prompt="Usted es un asistente experto en recomendaciones de recursos educativos y académicos."
    "Identifique las necesidades de aprendizaje del usuario y proporcione recursos adecuados."
    "Proporcione recomendaciones personalizadas basadas en las preferencias del usuario."
    "Considere libros, artículos, páginas web, videos y otros recursos relevantes para el aprendizaje."
    "Si el usuario solicita recomendaciones de artículos académicos, utilice la herramienta de búsqueda de Arxiv para encontrar artículos relevantes."
    "Utilice el lenguaje adecuado para un estudiante de secundaria.",
    name="recommender_assistant",
    checkpointer=MemorySaver(),
)


supervisor = create_supervisor(
    agents=[
        math_assistent,
        history_assistant,
        spanish_assistant,
        science_assistant,
        recommender_assistant,
        rag_agent,
    ],
    model=ChatOpenAI(model="gpt-4o"),
    prompt="Usted es un supervisor de agentes para un tutor académico especializado."
    "Supervise las respuestas de los agentes y dirija al usuario al agente adecuado según la pregunta."
    "Si la pregunta es sobre matemáticas, dirija al agente de matemáticas."
    "Si la pregunta es sobre ciencias, dirija al agente de ciencias."
    "Si la pregunta es sobre historia, dirija al agente de historia."
    "Si la pregunta es sobre español, gramática, ortografía, literatura, dirija al agente de español."
    "Si la pregunta es sobre recomendaciones de recursos educativos, dirija al agente de recomendaciones."
    "Si la pregunta es sobre un documento PDF o incluye un URL o enlace a un sitio web, dirija al agente de RAG (Retrieval Augmented Generation)."
    "Si la pregunta no es clara o no se puede responder, pida al usuario que aclare su pregunta."
    "Utilice el lenguaje adecuado para un estudiante de secundaria.",
).compile(checkpointer=MemorySaver())


config = {"configurable": {"thread_id": "abc123"}}


def stream_graph_updates(user_input: str):
    global config
    for event in supervisor.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",
        config=config,
    ):
        event["messages"][-1].pretty_print()
        print("\n")


if __name__ == "__main__":
    img = supervisor.get_graph().draw_mermaid_png()
    with open("images/supervisor_graph.png", "wb") as f:
        f.write(img)
    img = math_assistent.get_graph().draw_mermaid_png()
    with open("images/math_assistent_graph.png", "wb") as f:
        f.write(img)
    img = history_assistant.get_graph().draw_mermaid_png()
    with open("images/history_assistant_graph.png", "wb") as f:
        f.write(img)
    img = spanish_assistant.get_graph().draw_mermaid_png()
    with open("images/spanish_assistant_graph.png", "wb") as f:
        f.write(img)
    img = science_assistant.get_graph().draw_mermaid_png()
    with open("images/science_assistant_graph.png", "wb") as f:
        f.write(img)
    img = recommender_assistant.get_graph().draw_mermaid_png()
    with open("images/recommender_assistant_graph.png", "wb") as f:
        f.write(img)
    img = rag_agent.get_graph().draw_mermaid_png()
    with open("images/rag_agent_graph.png", "wb") as f:
        f.write(img)
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except:
            user_input = input()
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
