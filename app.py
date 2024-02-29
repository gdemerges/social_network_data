from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/statistiques')
def statistiques():
    with open('chemin/vers/votre/fichier/messages.json', 'r') as f:
        messages = json.load(f)
    stats = calculer_statistiques(messages)
    return jsonify(stats)

def calculer_statistiques(data):
    # Calcul de statistiques basiques, par exemple, le nombre total de messages
    nombre_messages = len(data['messages'])
    return {
        'nombre_messages': nombre_messages
    }

if __name__ == '__main__':
    app.run(debug=True)
