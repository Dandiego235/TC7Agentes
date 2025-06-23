from flask import Flask, render_template, request, jsonify
from agent import supervisor
from markdown import markdown
from langchain_core.messages import HumanMessage
from pprint import pprint

app = Flask(__name__)
config = {"configurable": {"thread_id": "abc123"}}

responses = []


@app.route("/")
def index():
    return render_template("chat.html") # Cargar la página


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
            pprint(event)
            print("\n")
            events.append(event)
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
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
