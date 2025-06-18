from flask import Flask, render_template, request, jsonify
from agent import supervisor

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")  # Render the chat interface

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        responses = []
        for event in supervisor.stream(
            {"messages": [{"role": "user", "content": user_input}]}
        ):
            for value in event.values():
                responses.append(value["messages"][-1].content)
        return jsonify({"responses": responses})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
