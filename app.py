from flask import Flask, render_template
import json

app = Flask(__name__)

@app.route('/')
def index():
    with open('message_1.json', 'r') as f:
        data = json.load(f)

    stats = calculer_statistiques(data)
    return render_template('index.html', stats=stats)

def calculer_statistiques(data):
    nombre_messages = len(data['messages'])
    return {
        'nombre_messages': nombre_messages
    }

if __name__ == '__main__':
    app.run(debug=True)
