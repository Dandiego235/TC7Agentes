from flask import Flask, render_template, request, jsonify
from agent import supervisor
from rag import process_pdf
from markdown import markdown
from langchain_core.messages import HumanMessage
from pprint import pprint

app = Flask(__name__)
config = {"configurable": {"thread_id": "abc123"}}

responses = []


@app.route("/")
def index():
    return render_template("chat.html")  # Cargar la página


@app.route("/ingest_pdf", methods=["POST"])
def ingest_pdf():
    data = request.json
    if not data or "file_path" not in data:
        return jsonify({"error": "Falta la ruta del archivo."}), 400

    file_path = data["file_path"]
    if file_path[0] == '"' and file_path[-1] == '"':
        file_path = file_path[1:-1]
    try:
        process_pdf(file_path)  # Asegurarse de que la ruta sea compatible con Unix
        return jsonify({"message": f"PDF procesado: {file_path}"})
    except Exception as e:
        print(f"Error procesando PDF {file_path}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No se brindó un mensaje"}), 400

    try:
        events = []
        responses = []
        for event in supervisor.stream(
            {"messages": [{"role": "user", "content": user_input}]}, config
        ):
            print(event)
            print("\n")
            events.append(event)
        print("Eventos de la conversación:")
        event = events[-1] if events else None
        for value in event.values():
            for message in value["messages"]:
                if isinstance(message, HumanMessage):
                    responses.append(
                        {
                            "role": "user-message",
                            "message": "Usuario: " + markdown(message.text()),
                        }
                    )
                else:
                    if markdown(message.text()) != "":
                        responses.append(
                            {
                                "role": "assistant-message",
                                "message": "Asistente: " + markdown(message.text()),
                            }
                        )

        return jsonify({"responses": responses})
    except Exception as e:
        print(f"Error en la conversación: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
