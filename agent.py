from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_tavily import TavilySearch
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


tools_list = load_tools(["arxiv"])
tavily_search_tool = TavilySearch(max_results=3)
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

math_assistent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    prompt="Usted es un asistente experto en matemáticas."
    "Responda a las preguntas de los usuarios de manera clara y concisa."
    "Explique el paso a paso de cómo resolver los problemas matemáticos.",
    name="math_assistant",
    checkpointer=MemorySaver(),
)

history_assistant = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[tavily_search_tool, wikipedia_tool],
    prompt="Usted es un asistente experto en historia."
    "Responda a las preguntas de los usuarios de manera clara y concisa."
    "Explique el contexto histórico y los eventos relevantes."
    "Utilice la búsqueda en línea para proporcionar información actualizada y precisa.",
    name="history_assistant",
    checkpointer=MemorySaver(),
)

science_assistant = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[tavily_search_tool, wikipedia_tool] + tools_list,
    prompt="Usted es un asistente experto en ciencias."
    "Responda a las preguntas de los usuarios de manera clara y concisa."
    "Explique el concepto científico y los principios relacionados.",
    name="science_assistant",
    checkpointer=MemorySaver(),
)

spanish_assistant = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    prompt="Usted es un asistente experto en español."
    "Responda a las preguntas de los usuarios de manera clara y concisa."
    "Revisa la redacción y la gramática de los textos en español, corrigiendo errores y mejorando la claridad.",
    name="spanish_assistant",
    checkpointer=MemorySaver(),
)

recommender_assistant = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[tavily_search_tool, wikipedia_tool] + tools_list,
    prompt="Usted es un asistente experto en recomendaciones de recursos educativos y académicos."
    "Identifique las necesidades de aprendizaje del usuario y proporcione recursos adecuados."
    "Proporcione recomendaciones personalizadas basadas en las preferencias del usuario."
    "Considere libros, artículos, páginas web, videos y otros recursos relevantes para el aprendizaje.",
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
    ],
    model=ChatOpenAI(model="gpt-4o-mini"),
    prompt="Usted es un supervisor de agentes de para un tutor académico especializado."
    "Supervise las respuestas de los agentes y dirija al usuario al agente adecuado según la pregunta."
    "Si la pregunta es sobre matemáticas, dirija al agente de matemáticas."
    "Si la pregunta es sobre ciencias, dirija al agente de ciencias."
    "Si la pregunta es sobre historia, dirija al agente de historia."
    "Si la pregunta es sobre español, dirija al agente de español."
    "Si la pregunta es sobre recomendaciones de recursos educativos, dirija al agente de recomendaciones."
    "Si la pregunta no es clara o no se puede responder, pida al usuario que aclare su pregunta.",
).compile(checkpointer=MemorySaver())


config = {"configurable": {"thread_id": "abc123"}}


def stream_graph_updates(user_input: str):
    global config
    for event in supervisor.stream(
        {"messages": [{"role": "user", "content": user_input}]}, config
    ):
        print(event)
        print("\n")

if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = input()  # "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
