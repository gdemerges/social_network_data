import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import json
from textblob import TextBlob
from datetime import datetime
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'])

def load_data():
    with open('message_1.json', 'r', encoding='UTF-8') as file:
        data = json.load(file)
    return data

data = load_data()

messages = pd.DataFrame(data['messages'])

nombre_total_messages = len(messages)

messages['date'] = pd.to_datetime(messages['timestamp_ms'], unit='ms')
messages_by_day = messages.groupby(messages['date'].dt.date).size()

def create_graph_component():
    return dcc.Graph(
        config={
            'displayModeBar': False
        }
    )

fig = px.line(x=messages_by_day.index, y=messages_by_day.values, labels={'x': 'Date', 'y': 'Nombre de messages'})
fig.update_layout(title='Nombre de Messages par Jour')

app.layout = html.Div([
    html.H1('Statistiques des Messages', className='text-center'),
    html.Div([
        html.H3(f'Nombre Total de Messages: {nombre_total_messages}', className='text-center')
    ], style={'margin': '20px'}),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=messages['date'].min(),
        end_date=messages['date'].max(),
        display_format='YYYY-MM-DD'
    ),
    dcc.Graph(
        id='graph-messages-by-day',
    config={
        'displayModeBar': False
    }
)
])

@app.callback(
    Output('graph-messages-by-day', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graph(start_date, end_date):
    filtered_messages = messages[(messages['date'] >= start_date) & (messages['date'] <= end_date)]
    messages_by_day = filtered_messages.groupby(filtered_messages['date'].dt.date).size()
    fig = px.line(x=messages_by_day.index, y=messages_by_day.values, labels={'x': 'Date', 'y': 'Nombre de messages'})
    fig.update_layout(title='Nombre de Messages par Jour')
    return fig

# Calcul des sentiments
messages['sentiment'] = messages['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Graphique des sentiments
@app.callback(
    Output('sentiment-graph', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_sentiment_graph(start_date, end_date):
    filtered_data = messages[(messages['date'] >= start_date) & (messages['date'] <= end_date)]
    sentiment_by_day = filtered_data.groupby(filtered_data['date'].dt.date)['sentiment'].mean()
    fig = px.line(x=sentiment_by_day.index, y=sentiment_by_day.values, labels={'x': 'Date', 'y': 'Sentiment moyen'})
    fig.update_layout(title='Sentiment Moyen des Messages par Jour')
    return fig

app.layout.children.append(dcc.Graph(id='sentiment-graph', config={
        'displayModeBar': False
    }))

@app.callback(
    Output('sender-histogram', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_sender_histogram(start_date, end_date):
    filtered_data = messages[(messages['date'] >= start_date) & (messages['date'] <= end_date)]
    sender_counts = filtered_data['sender_name'].value_counts()
    fig = px.bar(x=sender_counts.index, y=sender_counts.values, labels={'x': 'Expéditeur', 'y': 'Nombre de messages'})
    fig.update_layout(title='Messages par Expéditeur')
    return fig

app.layout.children.append(dcc.Graph(id='sender-histogram', config={
        'displayModeBar': False}))

if __name__ == '__main__':
    app.run_server(debug=True)
