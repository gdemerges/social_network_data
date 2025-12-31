"""
Composants du layout du dashboard - Design moderne dark mode.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

from .styles import (
    COLORS, PAGE_STYLE, HEADER_STYLE, HEADER_TITLE_STYLE, HEADER_SUBTITLE_STYLE,
    CARD_STYLE, CARD_HEADER_STYLE, UPLOAD_ZONE_STYLE, UPLOAD_ICON_STYLE,
    UPLOAD_TEXT_STYLE, UPLOAD_SUBTEXT_STYLE, GRAPH_CARD_STYLE,
    CHAT_CONTAINER_STYLE, CHAT_INPUT_STYLE, CHAT_BUTTON_STYLE,
    STAT_CARD_STYLE, STAT_VALUE_STYLE, STAT_LABEL_STYLE, EXPORT_BUTTON_STYLE
)


def create_header() -> html.Div:
    """Crée le header avec gradient."""
    return html.Div([
        html.H1("💬 Message Analyzer", style=HEADER_TITLE_STYLE),
        html.P("Analysez vos conversations avec l'IA • RAG + LLM Local", style=HEADER_SUBTITLE_STYLE),
    ], style=HEADER_STYLE)


def create_stats_cards() -> dbc.Row:
    """Crée les cartes de statistiques."""
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div(id='stat-messages', children="—", style=STAT_VALUE_STYLE),
                html.Div("Messages", style=STAT_LABEL_STYLE),
            ], style=STAT_CARD_STYLE)
        ], width=6, lg=3, className='mb-3'),
        dbc.Col([
            html.Div([
                html.Div(id='stat-participants', children="—", style=STAT_VALUE_STYLE),
                html.Div("Participants", style=STAT_LABEL_STYLE),
            ], style=STAT_CARD_STYLE)
        ], width=6, lg=3, className='mb-3'),
        dbc.Col([
            html.Div([
                html.Div(id='stat-chunks', children="—", style=STAT_VALUE_STYLE),
                html.Div("Chunks RAG", style=STAT_LABEL_STYLE),
            ], style=STAT_CARD_STYLE)
        ], width=6, lg=3, className='mb-3'),
        dbc.Col([
            html.Div([
                html.Div(id='stat-sentiment', children="—", style=STAT_VALUE_STYLE),
                html.Div("Sentiment moyen", style=STAT_LABEL_STYLE),
            ], style=STAT_CARD_STYLE)
        ], width=6, lg=3, className='mb-3'),
    ], className='mb-4')


def create_upload_section() -> html.Div:
    """Crée la section d'upload de fichiers."""
    return html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.Div("📂", style=UPLOAD_ICON_STYLE),
                html.Div("Glissez votre fichier ici (JSON, CSV, TXT)", style=UPLOAD_TEXT_STYLE),
                html.Div("ou cliquez pour sélectionner", style=UPLOAD_SUBTEXT_STYLE),
            ]),
            style=UPLOAD_ZONE_STYLE,
            multiple=False,
            accept='.json,.csv,.txt'
        ),
        dcc.Store(id='stored-data'),
        html.Div(id='output-data-upload'),
        html.Div(id='rag-status', className='mt-3'),
    ], style={**CARD_STYLE, 'marginBottom': '30px'})


def create_filters_section() -> html.Div:
    """Crée la section des filtres."""
    return html.Div([
        html.Div([
            html.Span("🔍", style={'marginRight': '10px'}),
            "Filtres"
        ], style=CARD_HEADER_STYLE),
        dbc.Row([
            dbc.Col([
                html.Label("Expéditeurs", style={'color': COLORS['text_muted'], 'marginBottom': '8px', 'fontSize': '0.9rem'}),
                dcc.Dropdown(
                    id='sender-dropdown',
                    multi=True,
                    placeholder='Tous les expéditeurs...',
                    style={'backgroundColor': COLORS['card_bg']},
                    className='dark-dropdown'
                ),
            ], width=12, lg=6, className='mb-3'),
            dbc.Col([
                html.Label("Période", style={'color': COLORS['text_muted'], 'marginBottom': '8px', 'fontSize': '0.9rem'}),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    display_format='DD/MM/YYYY',
                    start_date_placeholder_text='Date début',
                    end_date_placeholder_text='Date fin',
                    className='dark-datepicker'
                ),
            ], width=12, lg=6, className='mb-3'),
        ]),
    ], style=CARD_STYLE)


def create_graphs_section() -> dbc.Row:
    """Crée la section des graphiques."""
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.Span("📈", style={'marginRight': '10px'}),
                    "Messages par jour"
                ], style={**CARD_HEADER_STYLE, 'marginBottom': '15px'}),
                dcc.Graph(
                    id='graph-messages-by-day',
                    config={'displayModeBar': False},
                    style={'height': '250px'}
                ),
            ], style=GRAPH_CARD_STYLE)
        ], width=12, lg=4, className='mb-4'),
        dbc.Col([
            html.Div([
                html.Div([
                    html.Span("😊", style={'marginRight': '10px'}),
                    "Sentiment"
                ], style={**CARD_HEADER_STYLE, 'marginBottom': '15px'}),
                dcc.Graph(
                    id='sentiment-graph',
                    config={'displayModeBar': False},
                    style={'height': '250px'}
                ),
            ], style=GRAPH_CARD_STYLE)
        ], width=12, lg=4, className='mb-4'),
        dbc.Col([
            html.Div([
                html.Div([
                    html.Span("👥", style={'marginRight': '10px'}),
                    "Par expéditeur"
                ], style={**CARD_HEADER_STYLE, 'marginBottom': '15px'}),
                dcc.Graph(
                    id='sender-histogram',
                    config={'displayModeBar': False},
                    style={'height': '250px'}
                ),
            ], style=GRAPH_CARD_STYLE)
        ], width=12, lg=4, className='mb-4'),
    ])


def create_analytics_section() -> html.Div:
    """Crée la section d'analytics avancées."""
    return html.Div([
        # Header avec toggle
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("📊", style={'marginRight': '10px', 'fontSize': '1.5rem'}),
                    html.Span("Analytics Avancées", style={'fontWeight': '600', 'fontSize': '1.2rem', 'color': COLORS['text']}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ]),
            dbc.Col([
                dbc.Button(
                    "Afficher/Masquer",
                    id="toggle-analytics",
                    size="sm",
                    style={'backgroundColor': COLORS['card_hover'], 'border': 'none'}
                )
            ], width='auto'),
        ], className='mb-3', align='center'),

        # Zone analytics (collapsible)
        dbc.Collapse(
            id="analytics-collapse",
            is_open=False,
            children=[
                dbc.Row([
                    # Word Cloud
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.Span("☁️", style={'marginRight': '10px'}),
                                "Mots les plus fréquents"
                            ], style={**CARD_HEADER_STYLE, 'marginBottom': '15px'}),
                            dcc.Graph(
                                id='wordcloud-graph',
                                config={'displayModeBar': False},
                                style={'height': '300px'}
                            ),
                        ], style=GRAPH_CARD_STYLE)
                    ], width=12, lg=6, className='mb-4'),

                    # Topics
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.Span("🏷️", style={'marginRight': '10px'}),
                                "Topics détectés"
                            ], style={**CARD_HEADER_STYLE, 'marginBottom': '15px'}),
                            dcc.Graph(
                                id='topics-graph',
                                config={'displayModeBar': False},
                                style={'height': '300px'}
                            ),
                        ], style=GRAPH_CARD_STYLE)
                    ], width=12, lg=6, className='mb-4'),
                ]),

                dbc.Row([
                    # Network Graph
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.Span("🕸️", style={'marginRight': '10px'}),
                                "Réseau d'interactions"
                            ], style={**CARD_HEADER_STYLE, 'marginBottom': '15px'}),
                            dcc.Graph(
                                id='network-graph',
                                config={'displayModeBar': False},
                                style={'height': '400px'}
                            ),
                        ], style=GRAPH_CARD_STYLE)
                    ], width=12, lg=6, className='mb-4'),

                    # Activity Heatmap
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.Span("🔥", style={'marginRight': '10px'}),
                                "Activité (Jour × Heure)"
                            ], style={**CARD_HEADER_STYLE, 'marginBottom': '15px'}),
                            dcc.Graph(
                                id='heatmap-graph',
                                config={'displayModeBar': False},
                                style={'height': '400px'}
                            ),
                        ], style=GRAPH_CARD_STYLE)
                    ], width=12, lg=6, className='mb-4'),
                ]),
            ]
        ),
    ], style=CARD_STYLE)


def create_chat_section() -> html.Div:
    """Crée la section de chat avec l'IA."""
    return html.Div([
        # Header du chat
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("🤖", style={'marginRight': '10px', 'fontSize': '1.5rem'}),
                    html.Span("Assistant IA", style={'fontWeight': '600', 'fontSize': '1.2rem', 'color': COLORS['text']}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ]),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='model-selector',
                            options=[
                                {'label': '🚀 Mistral 7B', 'value': 'mistral'},
                                {'label': '🦙 Llama 3 8B', 'value': 'llama3'},
                                {'label': '🔬 Phi-3 3.8B', 'value': 'phi3'},
                                {'label': '💎 Gemma 7B', 'value': 'gemma'},
                            ],
                            value='mistral',
                            clearable=False,
                            style={'width': '180px'},
                            className='dark-dropdown'
                        ),
                    ], width='auto'),
                    dbc.Col([
                        html.Div(id='ollama-status'),
                    ], width='auto', style={'display': 'flex', 'alignItems': 'center'}),
                ], justify='end', align='center'),
            ]),
        ], className='mb-3', align='center'),
        
        # Zone de chat
        html.Div(
            id='chat-history',
            style=CHAT_CONTAINER_STYLE,
            children=[
                html.Div([
                    html.Div("👋", style={'fontSize': '3rem', 'marginBottom': '15px'}),
                    html.P("Bienvenue !", style={'color': COLORS['text'], 'fontSize': '1.2rem', 'fontWeight': '600', 'marginBottom': '10px'}),
                    html.P(
                        "Chargez un fichier JSON puis posez vos questions sur vos messages.",
                        style={'color': COLORS['text_muted'], 'maxWidth': '400px', 'margin': '0 auto'}
                    ),
                ], style={'textAlign': 'center', 'padding': '60px 20px'})
            ]
        ),
        
        # Zone de saisie
        dbc.InputGroup([
            dbc.Input(
                id='chat-input',
                placeholder='Posez une question sur vos messages...',
                type='text',
                style=CHAT_INPUT_STYLE
            ),
            dbc.Button(
                [html.Span("Envoyer"), html.Span(" →", style={'marginLeft': '8px'})],
                id='send-button',
                style=CHAT_BUTTON_STYLE
            ),
        ]),
        
        # Loading
        dcc.Loading(
            id='loading-chat',
            type='dot',
            color=COLORS['primary'],
            children=html.Div(id='chat-loading-output')
        ),
    ], style=CARD_STYLE)


def create_config_section() -> html.Div:
    """Crée la section de configuration RAG."""
    return html.Div([
        # Header avec toggle
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("⚙️", style={'marginRight': '10px', 'fontSize': '1.3rem'}),
                    html.Span("Configuration RAG", style={'fontWeight': '600', 'fontSize': '1.1rem', 'color': COLORS['text']}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ]),
            dbc.Col([
                dbc.Button(
                    "Afficher/Masquer",
                    id="toggle-config",
                    size="sm",
                    style={'backgroundColor': COLORS['card_hover'], 'border': 'none'}
                )
            ], width='auto'),
        ], className='mb-3', align='center'),

        # Zone de configuration (collapsible)
        dbc.Collapse(
            id="config-collapse",
            is_open=False,
            children=[
                dbc.Row([
                    # Cache
                    dbc.Col([
                        html.Label("💾 Cache", style={'color': COLORS['text'], 'fontWeight': '500', 'marginBottom': '8px'}),
                        dbc.Checklist(
                            id='config-cache-enabled',
                            options=[{'label': ' Activer le cache (TTL: 1h)', 'value': 'enabled'}],
                            value=['enabled'],
                            switch=True,
                            style={'color': COLORS['text']}
                        ),
                    ], width=12, lg=6, className='mb-3'),

                    # Nombre de contextes
                    dbc.Col([
                        html.Label("📄 Nombre de contextes", style={'color': COLORS['text'], 'fontWeight': '500', 'marginBottom': '8px'}),
                        dcc.Slider(
                            id='config-n-context',
                            min=1,
                            max=10,
                            step=1,
                            value=5,
                            marks={i: str(i) for i in range(1, 11)},
                            tooltip={"placement": "bottom", "always_visible": False}
                        ),
                    ], width=12, lg=6, className='mb-3'),
                ]),

                dbc.Row([
                    # Recherche hybride
                    dbc.Col([
                        html.Label("🔍 Recherche Hybride", style={'color': COLORS['text'], 'fontWeight': '500', 'marginBottom': '8px'}),
                        dbc.Checklist(
                            id='config-hybrid-search',
                            options=[{'label': ' Vector + BM25', 'value': 'enabled'}],
                            value=['enabled'],
                            switch=True,
                            style={'color': COLORS['text']}
                        ),
                    ], width=12, lg=6, className='mb-3'),

                    # Re-ranking
                    dbc.Col([
                        html.Label("🎯 Re-ranking", style={'color': COLORS['text'], 'fontWeight': '500', 'marginBottom': '8px'}),
                        dbc.Checklist(
                            id='config-reranking',
                            options=[{'label': ' Cross-encoder', 'value': 'enabled'}],
                            value=['enabled'],
                            switch=True,
                            style={'color': COLORS['text']}
                        ),
                    ], width=12, lg=6, className='mb-3'),
                ]),

                # Stats du cache
                html.Div(id='cache-stats-display', className='mt-3'),
            ]
        ),
    ], style=CARD_STYLE)


def create_export_section() -> html.Div:
    """Crée la section d'export."""
    return html.Div([
        html.Button(
            [html.Span("📥", style={'marginRight': '10px'}), "Exporter les données"],
            id='export-button',
            style=EXPORT_BUTTON_STYLE,
            className='export-btn'
        ),
        dcc.Download(id='download-data'),
    ], style={'textAlign': 'center', 'padding': '20px 0'})


def create_layout() -> html.Div:
    """Crée le layout complet du dashboard."""
    return html.Div([
        dbc.Container([
            create_header(),
            create_upload_section(),
            create_stats_cards(),
            create_filters_section(),
            html.Div(style={'height': '30px'}),
            create_graphs_section(),
            html.Div(style={'height': '30px'}),
            create_analytics_section(),
            html.Div(style={'height': '30px'}),
            create_chat_section(),
            html.Div(style={'height': '20px'}),
            create_config_section(),
            create_export_section(),
            
            # Stores
            dcc.Store(id='chat-history-store', data=[]),
            dcc.Store(id='streaming-session-id', data=None),
            dcc.Store(id='streaming-active', data=False),

            # Interval pour le streaming (désactivé par défaut)
            dcc.Interval(
                id='streaming-interval',
                interval=200,  # Mise à jour toutes les 200ms
                n_intervals=0,
                disabled=True  # Désactivé par défaut
            ),

        ], fluid=True),
    ], style=PAGE_STYLE)
