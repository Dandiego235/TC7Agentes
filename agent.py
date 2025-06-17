from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor


math_assistent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    prompt="Usted es un asistente experto en matemáticas."
    "Responda a las preguntas de los usuarios de manera clara y concisa."
    "Explique el paso a paso de cómo resolver los problemas matemáticos.",
    name="math_assistant",
)

history_assistant = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    prompt="Usted es un asistente experto en historia."
    "Responda a las preguntas de los usuarios de manera clara y concisa."
    "Explique el contexto histórico y los eventos relevantes.",
    name="history_assistant",
)

spanish_assistant = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    prompt="Usted es un asistente experto en español."
    "Responda a las preguntas de los usuarios de manera clara y concisa."
    "Revisa la redacción y la gramática de los textos en español, corrigiendo errores y mejorando la claridad.",
    name="spanish_assistant",
)

recommender_assistant = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    prompt="Usted es un asistente experto en recomendaciones de recursos educativos y académicos."
    "Proporcione recomendaciones personalizadas basadas en las preferencias del usuario."
    "Considere libros, artículos, páginas web, videos y otros recursos relevantes para el aprendizaje.",
    name="recommender_assistant",
)

supervisor = create_supervisor(
    agents=[
        math_assistent,
        history_assistant,
        spanish_assistant,
        recommender_assistant,
    ],
    model=ChatOpenAI(model="gpt-4o-mini"),
    prompt="Usted es un supervisor de agentes de para un tutor académico especializado."
    "Supervise las respuestas de los agentes y dirija al usuario al agente adecuado según la pregunta."
    "Si la pregunta es sobre matemáticas, dirija al agente de matemáticas."
    "Si la pregunta es sobre historia, dirija al agente de historia."
    "Si la pregunta es sobre español, dirija al agente de español."
    "Si la pregunta es sobre recomendaciones de recursos educativos, dirija al agente de recomendaciones."
    "Si la pregunta no es clara o no se puede responder, pida al usuario que aclare su pregunta.",
).compile()


def stream_graph_updates(user_input: str):
    for event in supervisor.stream(
        {"messages": [{"role": "user", "content": user_input}]}
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


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
