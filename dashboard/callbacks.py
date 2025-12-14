"""
Callbacks du dashboard Dash - Version Dark Theme.
"""

from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

from .data_processing import decode_upload_content, process_messages, filter_messages, compute_statistics
from .styles import COLORS, PLOTLY_LAYOUT, PLOTLY_COLORS
from rag import get_rag_engine


def register_callbacks(app):
    """Enregistre tous les callbacks de l'application."""
    
    register_data_callback(app)
    register_model_callback(app)
    register_chat_callback(app)
    register_export_callback(app)


def apply_dark_theme(fig):
    """Applique le thème sombre à un graphique Plotly."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def register_data_callback(app):
    """Callback pour le chargement et filtrage des données."""
    
    @app.callback(
        [Output('graph-messages-by-day', 'figure'),
         Output('sentiment-graph', 'figure'),
         Output('sender-histogram', 'figure'),
         Output('sender-dropdown', 'options'),
         Output('stored-data', 'data'),
         Output('rag-status', 'children'),
         Output('stat-messages', 'children'),
         Output('stat-participants', 'children'),
         Output('stat-chunks', 'children'),
         Output('stat-sentiment', 'children')],
        [Input('upload-data', 'contents'),
         Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date'),
         Input('sender-dropdown', 'value')],
        State('upload-data', 'filename')
    )
    def update_output(content, start_date, end_date, selected_senders, filename):
        if content is None:
            raise PreventUpdate

        # Traiter les données
        data = decode_upload_content(content, filename)
        messages = process_messages(data)
        
        # Indexer dans le RAG
        rag = get_rag_engine()
        indexed_count = rag.index_messages(messages)
        
        # Statistiques RAG
        _ = rag.get_stats()  # Pour vérification interne
        rag_status = html.Div([
            dbc.Badge("✅ RAG Actif", color="success", className="me-2"),
            html.Small(f"{indexed_count} chunks indexés", style={'color': COLORS['text_muted']})
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'})

        # Filtrer les messages
        filtered = filter_messages(messages, start_date, end_date, selected_senders)
        
        # Calculer les stats
        stats_data = compute_statistics(filtered)
        
        # Options du dropdown
        sender_options = [
            {'label': f'👤 {sender}', 'value': sender}
            for sender in stats_data['unique_senders']
        ]

        # Créer les graphiques avec thème sombre
        fig1 = px.area(
            stats_data['messages_by_day'],
            labels={'value': 'Messages', 'index': ''},
        )
        fig1.update_traces(
            fill='tozeroy',
            line=dict(color=PLOTLY_COLORS[0], width=2),
            fillcolor='rgba(99, 102, 241, 0.2)'
        )
        apply_dark_theme(fig1)
        fig1.update_layout(showlegend=False, title='')
        
        # Sentiment graph
        fig2 = px.line(
            stats_data['sentiment_by_day'],
            labels={'value': 'Sentiment', 'index': ''},
        )
        fig2.update_traces(line=dict(color=PLOTLY_COLORS[1], width=2))
        apply_dark_theme(fig2)
        fig2.update_layout(showlegend=False, title='')
        
        # Sender histogram
        fig3 = px.bar(
            stats_data['sender_counts'],
            labels={'value': 'Messages', 'index': ''},
        )
        fig3.update_traces(marker_color=PLOTLY_COLORS[2])
        apply_dark_theme(fig3)
        fig3.update_layout(showlegend=False, title='')

        # Stats cards values
        total_messages = len(filtered)
        total_participants = len(stats_data['unique_senders'])
        avg_sentiment = round(stats_data['sentiment_by_day'].mean(), 2) if not stats_data['sentiment_by_day'].empty else 0
        sentiment_emoji = "😊" if avg_sentiment > 0.1 else ("😐" if avg_sentiment > -0.1 else "😞")

        return (
            fig1, fig2, fig3,
            sender_options,
            filtered.to_json(date_format='iso', orient='split'),
            rag_status,
            f"{total_messages:,}".replace(",", " "),
            str(total_participants),
            str(indexed_count),
            f"{sentiment_emoji} {avg_sentiment}"
        )


def register_model_callback(app):
    """Callback pour la sélection du modèle Ollama."""
    
    @app.callback(
        Output('ollama-status', 'children'),
        [Input('model-selector', 'value')]
    )
    def update_model(model_name):
        rag = get_rag_engine()
        rag.ollama_model = model_name
        
        status = rag.check_ollama_status()
        
        if status['status'] == 'online':
            if status['model_available']:
                return html.Div([
                    html.Span("●", style={'color': COLORS['secondary'], 'marginRight': '6px', 'fontSize': '0.8rem'}),
                    html.Span("En ligne", style={'color': COLORS['secondary'], 'fontSize': '0.85rem'})
                ], style={'display': 'flex', 'alignItems': 'center'})
            else:
                return html.Div([
                    html.Span("●", style={'color': COLORS['accent'], 'marginRight': '6px', 'fontSize': '0.8rem'}),
                    html.Span("Modèle manquant", style={'color': COLORS['accent'], 'fontSize': '0.85rem'})
                ], style={'display': 'flex', 'alignItems': 'center'})
        else:
            return html.Div([
                html.Span("●", style={'color': '#ef4444', 'marginRight': '6px', 'fontSize': '0.8rem'}),
                html.Span("Hors ligne", style={'color': '#ef4444', 'fontSize': '0.85rem'})
            ], style={'display': 'flex', 'alignItems': 'center'})


def register_chat_callback(app):
    """Callback pour le chat avec l'IA."""
    
    @app.callback(
        [Output('chat-history', 'children'),
         Output('chat-history-store', 'data'),
         Output('chat-input', 'value'),
         Output('chat-loading-output', 'children')],
        [Input('send-button', 'n_clicks'),
         Input('chat-input', 'n_submit')],
        [State('chat-input', 'value'),
         State('chat-history-store', 'data'),
         State('stored-data', 'data')],
        prevent_initial_call=True
    )
    def handle_chat(n_clicks, n_submit, user_input, chat_history, stored_data):
        if not user_input or not user_input.strip():
            raise PreventUpdate
        
        if chat_history is None:
            chat_history = []
        
        # Message utilisateur
        chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Générer la réponse
        if not stored_data:
            assistant_response = "⚠️ Veuillez d'abord charger un fichier JSON contenant vos messages."
        else:
            rag = get_rag_engine()
            result = rag.chat(user_input)
            assistant_response = result['answer']
            
            # Ajouter les sources (seulement pour les réponses RAG, pas statistiques)
            if result.get('sources') and result.get('retrieval_method') != 'statistical_analysis':
                sources_text = "\n\n---\n📚 **Extraits de conversation utilisés:**\n"
                for i, src in enumerate(result['sources'][:3], 1):
                    content = src.get('content', '')
                    metadata = src.get('metadata', {})
                    
                    # Pour les chunks de fenêtre, utiliser 'senders' au lieu de 'sender_name'
                    if metadata.get('chunk_type') == 'conversation_window':
                        senders = metadata.get('senders', 'Inconnu')
                        start_date = metadata.get('start_date', '')
                        end_date = metadata.get('end_date', '')
                        date_info = f"{start_date}" if start_date == end_date else f"{start_date} → {end_date}"
                        
                        # Limiter la longueur et nettoyer le contenu
                        content_preview = content[:200] if len(content) > 200 else content
                        content_preview = content_preview.replace('\n', ' • ')
                        
                        sources_text += f"\n**{i}. Conversation** ({date_info})\n"
                        sources_text += f"   Participants: {senders}\n"
                        sources_text += f"   > {content_preview}{'...' if len(content) > 200 else ''}\n"
                    else:
                        # Messages individuels
                        sender = metadata.get('sender_name', 'Inconnu')
                        date = metadata.get('date', '')
                        content_preview = content[:150] if len(content) > 150 else content
                        
                        sources_text += f"\n**{i}. [{sender}]** ({date})\n"
                        sources_text += f"   > {content_preview}{'...' if len(content) > 150 else ''}\n"
                
                if len(result['sources']) > 0:
                    assistant_response += sources_text
        
        # Réponse assistant
        chat_history.append({
            'role': 'assistant',
            'content': assistant_response
        })
        
        # Construire l'affichage
        chat_display = build_chat_display(chat_history)
        
        return chat_display, chat_history, '', None


def build_chat_display(chat_history: list) -> list:
    """Construit l'affichage du chat avec le thème sombre."""
    chat_display = []
    
    for msg in chat_history:
        if msg['role'] == 'user':
            # Message utilisateur - aligné à droite, style accent
            chat_display.append(
                html.Div([
                    html.Div([
                        html.Div(msg['content'], style={
                            'backgroundColor': COLORS['primary'],
                            'color': 'white',
                            'padding': '12px 16px',
                            'borderRadius': '18px 18px 4px 18px',
                            'maxWidth': '75%',
                            'marginLeft': 'auto',
                            'fontSize': '0.95rem',
                            'lineHeight': '1.5',
                            'boxShadow': '0 2px 8px rgba(99, 102, 241, 0.3)'
                        })
                    ], style={'display': 'flex', 'justifyContent': 'flex-end'})
                ], className='chat-message', style={'marginBottom': '12px'})
            )
        else:
            # Message assistant - aligné à gauche, style carte
            chat_display.append(
                html.Div([
                    html.Div([
                        html.Span("🤖", style={'marginRight': '10px', 'fontSize': '1.2rem'}),
                        html.Div([
                            dcc.Markdown(
                                msg['content'],
                                style={
                                    'margin': 0,
                                    'color': COLORS['text'],
                                    'fontSize': '0.95rem',
                                    'lineHeight': '1.6'
                                }
                            )
                        ], style={
                            'backgroundColor': COLORS['card_hover'],
                            'padding': '12px 16px',
                            'borderRadius': '4px 18px 18px 18px',
                            'maxWidth': '75%',
                            'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.2)'
                        })
                    ], style={'display': 'flex', 'alignItems': 'flex-start'})
                ], className='chat-message', style={'marginBottom': '12px'})
            )
    
    return chat_display


def register_export_callback(app):
    """Callback pour l'export des données."""
    
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
