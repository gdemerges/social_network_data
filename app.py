from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

@app.route('/')
def index():
    try:
        input_encoding = 'Latin-1'

        with open('message_1.json', 'r', encoding=input_encoding) as file:
            data = json.load(file)

        with open('message_1_utf8.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    except FileNotFoundError:
        return "Fichier JSON non trouvé.", 404
    except json.JSONDecodeError:
        return "Erreur de décodage JSON.", 500

    try:
        stats = calculer_statistiques(data)
        stats_message = calcul_messages_by_people(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return render_template('index.html', stats=stats, stats_message=stats_message)

def calculer_statistiques(data):
    nombre_messages = len(data['messages'])
    return {
        'nombre_messages': nombre_messages
    }

def calcul_messages_by_people(data):
    message_count = {}
    for message in data["messages"]:
        sender = message["sender_name"]
        if sender in message_count:
            message_count[sender] += 1
        else:
            message_count[sender] = 1
    return message_count

if __name__ == '__main__':
    app.run(debug=True)
