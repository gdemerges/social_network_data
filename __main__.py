import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import json
from textblob import TextBlob
import dash_bootstrap_components as dbc
import base64
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def decode_content(content):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    return json.loads(decoded.decode('utf-8'))

def process_data(data):
    messages = pd.DataFrame(data['messages'])
    messages['date'] = pd.to_datetime(messages['timestamp_ms'], unit='ms')
    messages['sentiment'] = messages['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return messages

app.layout = dbc.Container([
    html.H1('Statistiques des Messages', className='text-center mb-4'),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Charger Fichier JSON', className='btn btn-primary btn-lg'),
        style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
        multiple=False
    ),
    dcc.Store(id='stored-data'),
    html.Div(id='output-data-upload'),
    dcc.Dropdown(id='sender-dropdown', multi=True, placeholder='Filtrer par expéditeur...'),
    dcc.DatePickerRange(id='date-picker-range', display_format='YYYY-MM-DD', start_date_placeholder_text='Start Date', end_date_placeholder_text='End Date'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='graph-messages-by-day', config={'displayModeBar': False}), width=12, lg=4),
        dbc.Col(dcc.Graph(id='sentiment-graph', config={'displayModeBar': False}), width=12, lg=4),
        dbc.Col(dcc.Graph(id='sender-histogram', config={'displayModeBar': False}), width=12, lg=4),
    ]),
    html.Button('Exporter Données', id='export-button', className='btn btn-success'),
    dcc.Download(id='download-data')
], fluid=True)

@app.callback(
    [Output('graph-messages-by-day', 'figure'),
     Output('sentiment-graph', 'figure'),
     Output('sender-histogram', 'figure'),
     Output('sender-dropdown', 'options'),
     Output('stored-data', 'data')],
    [Input('upload-data', 'contents'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('sender-dropdown', 'value')],
    State('upload-data', 'filename')
)
def update_output(content, start_date, end_date, selected_senders, filename):
    if content is None:
        raise PreventUpdate

    data = decode_content(content)
    messages = process_data(data)

    if start_date and end_date:
        mask = (messages['date'] >= start_date) & (messages['date'] <= end_date)
        messages = messages.loc[mask]

    if selected_senders:
        messages = messages[messages['sender_name'].isin(selected_senders)]

    messages_by_day = messages.groupby(messages['date'].dt.date).size()
    sentiment_by_day = messages.groupby(messages['date'].dt.date)['sentiment'].mean()
    sender_counts = messages['sender_name'].value_counts()

    sender_options = [{'label': sender, 'value': sender} for sender in messages['sender_name'].unique()]

    fig1 = px.line(messages_by_day, labels={'value': 'Nombre de messages', 'index': 'Date'}, title='Messages par Jour')
    fig2 = px.line(sentiment_by_day, labels={'value': 'Sentiment moyen', 'index': 'Date'}, title='Sentiment Moyen par Jour')
    fig3 = px.bar(sender_counts, labels={'value': 'Nombre de messages', 'index': 'Expéditeur'}, title='Messages par Expéditeur')

    return fig1, fig2, fig3, sender_options, messages.to_json(date_format='iso', orient='split')

@app.callback(
    Output('download-data', 'data'),
    [Input('export-button', 'n_clicks')],
    [State('stored-data', 'data')]
)
def export_data(n_clicks, stored_data):
    if n_clicks and stored_data:
        messages = pd.read_json(stored_data, orient='split')
        return dcc.send_data_frame(messages.to_csv, "exported_messages.csv")
    raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)
