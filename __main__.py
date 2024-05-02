import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import json
from textblob import TextBlob
from datetime import datetime
import dash_bootstrap_components as dbc
import base64
import io

app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'json' in filename:
            data = json.loads(decoded.decode('utf-8'))
            messages = pd.DataFrame(data['messages'])
            messages['date'] = pd.to_datetime(messages['timestamp_ms'], unit='ms')
            messages['sentiment'] = messages['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            return messages
    except Exception as e:
        print(e)
        return html.Div(['There was an error processing this file.'])

app.layout = html.Div([
    html.H1('Statistiques des Messages', className='text-center'),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Charger Fichier JSON', className='btn btn-primary'),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Div(id='total-messages', className='text-center'),
    dcc.DatePickerRange(
        id='date-picker-range',
        display_format='YYYY-MM-DD'
    ),
    dcc.Graph(id='graph-messages-by-day', config={'displayModeBar': False}),
    dcc.Graph(id='sentiment-graph', config={'displayModeBar': False}),
    dcc.Graph(id='sender-histogram', config={'displayModeBar': False})
])

@app.callback(
    [Output('graph-messages-by-day', 'figure'),
     Output('sentiment-graph', 'figure'),
     Output('sender-histogram', 'figure'),
     Output('date-picker-range', 'start_date'),
     Output('date-picker-range', 'end_date'),
     Output('total-messages', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents:
        messages = parse_contents(contents, filename)
        if messages is not None:
            total_messages = f"Nombre Total de Messages: {len(messages)}"
            start_date = messages['date'].min()
            end_date = messages['date'].max()
            messages_by_day = messages.groupby(messages['date'].dt.date).size()
            sentiment_by_day = messages.groupby(messages['date'].dt.date)['sentiment'].mean()
            sender_counts = messages['sender_name'].value_counts()

            fig1 = px.line(x=messages_by_day.index, y=messages_by_day.values, labels={'x': 'Date', 'y': 'Nombre de messages'})
            fig1.update_layout(title='Nombre de Messages par Jour')

            fig2 = px.line(x=sentiment_by_day.index, y=sentiment_by_day.values, labels={'x': 'Date', 'y': 'Sentiment moyen'})
            fig2.update_layout(title='Sentiment Moyen des Messages par Jour')

            fig3 = px.bar(x=sender_counts.index, y=sender_counts.values, labels={'x': 'Expéditeur', 'y': 'Nombre de messages'})
            fig3.update_layout(title='Messages par Expéditeur')

            return fig1, fig2, fig3, start_date, end_date, total_messages
    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
