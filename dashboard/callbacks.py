"""
Callbacks du dashboard Dash - Version Dark Theme.
"""

import json
import uuid
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from flask import request

from .data_processing import decode_upload_content, process_messages, filter_messages, compute_statistics
from .styles import COLORS, PLOTLY_LAYOUT, PLOTLY_COLORS
from rag import get_rag_engine
from rag.llm_client import get_stream_buffer
from rag.rate_limiter import get_rate_limiter
from analytics import get_analytics


def register_callbacks(app):
    """Enregistre tous les callbacks de l'application."""

    register_data_callback(app)
    register_model_callback(app)
    register_chat_callback(app)
    # register_streaming_callback(app)  # TODO: À implémenter (voir STREAMING_SETUP.md)
    register_config_callback(app)
    register_analytics_callback(app)
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

        try:
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

        except ValueError as e:
            # Erreur de validation (ex: taille de fichier)
            error_msg = str(e)
            return (
                {}, {}, {},  # Graphiques vides
                [],  # Options dropdown
                None,  # stored-data
                html.Div([
                    dbc.Badge("❌ Erreur", color="danger", className="me-2"),
                    html.Small(error_msg, style={'color': '#ef4444'})
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),
                "0", "0", "0", "N/A"  # Stats par défaut
            )
        except json.JSONDecodeError:
            error_msg = "❌ Format JSON invalide. Vérifiez que votre fichier est bien formaté."
            return (
                {}, {}, {},
                [],
                None,
                html.Div([
                    dbc.Badge("❌ Erreur", color="danger", className="me-2"),
                    html.Small(error_msg, style={'color': '#ef4444'})
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),
                "0", "0", "0", "N/A"
            )
        except Exception as e:
            # Erreur générique
            error_msg = f"❌ Erreur lors du traitement: {str(e)[:100]}"
            return (
                {}, {}, {},
                [],
                None,
                html.Div([
                    dbc.Badge("❌ Erreur", color="danger", className="me-2"),
                    html.Small(error_msg, style={'color': '#ef4444'})
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),
                "0", "0", "0", "N/A"
            )

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
        try:
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
        except Exception as e:
            return html.Div([
                html.Span("●", style={'color': '#ef4444', 'marginRight': '6px', 'fontSize': '0.8rem'}),
                html.Span(f"Erreur: {str(e)[:30]}", style={'color': '#ef4444', 'fontSize': '0.85rem'})
            ], style={'display': 'flex', 'alignItems': 'center'})


def register_chat_callback(app):
    """Callback pour le chat avec l'IA (avec streaming)."""

    @app.callback(
        [Output('chat-history', 'children'),
         Output('chat-history-store', 'data'),
         Output('chat-input', 'value'),
         Output('chat-loading-output', 'children'),
         Output('streaming-session-id', 'data'),
         Output('streaming-active', 'data'),
         Output('streaming-interval', 'disabled')],
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

        # Vérifier le rate limiting
        rate_limiter = get_rate_limiter()
        client_ip = request.remote_addr if request else "unknown"
        allowed, error_msg = rate_limiter.is_allowed(client_ip)

        if not allowed:
            # Rate limit dépassé
            chat_history.append({'role': 'user', 'content': user_input})
            chat_history.append({'role': 'assistant', 'content': error_msg})
            chat_display = build_chat_display(chat_history)
            return chat_display, chat_history, '', None

        # Message utilisateur
        chat_history.append({
            'role': 'user',
            'content': user_input
        })

        # Générer la réponse
        try:
            if not stored_data:
                assistant_response = "⚠️ Veuillez d'abord charger un fichier JSON contenant vos messages."
            else:
                rag = get_rag_engine()
                result = rag.chat(user_input)
                assistant_response = result['answer']

                # Enregistrer le succès pour le circuit breaker
                rate_limiter.record_success()

                # Indiquer si la réponse vient du cache
                if result.get('from_cache'):
                    assistant_response = "⚡ **(Réponse en cache - instantanée)**\n\n" + assistant_response

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

        except ConnectionError:
            # Enregistrer l'échec pour le circuit breaker
            rate_limiter.record_failure()
            assistant_response = "❌ **Erreur de connexion**\n\nImpossible de contacter le serveur Ollama. Vérifiez que :\n- Ollama est bien démarré (`ollama serve`)\n- L'URL est correcte (par défaut: http://localhost:11434)"
        except TimeoutError:
            rate_limiter.record_failure()
            assistant_response = "⏱️ **Timeout**\n\nLe modèle met trop de temps à répondre. Essayez :\n- Un modèle plus léger (phi3, gemma)\n- Une question plus simple"
        except Exception as e:
            rate_limiter.record_failure()
            error_type = type(e).__name__
            assistant_response = f"❌ **Erreur ({error_type})**\n\n{str(e)[:200]}\n\nVeuillez réessayer ou reformuler votre question."

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


def register_config_callback(app):
    """Callbacks pour la configuration RAG."""

    @app.callback(
        Output("config-collapse", "is_open"),
        [Input("toggle-config", "n_clicks")],
        [State("config-collapse", "is_open")],
    )
    def toggle_config(n, is_open):
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output('cache-stats-display', 'children'),
        [Input('config-cache-enabled', 'value'),
         Input('chat-history-store', 'data')]
    )
    def update_cache_stats(cache_enabled, chat_history):
        """Affiche les stats du cache."""
        rag = get_rag_engine()
        stats = rag.get_stats()

        if 'cache_stats' not in stats:
            return html.Div()

        cache_stats = stats['cache_stats']
        hit_rate = cache_stats.get('hit_rate', 0)
        hits = cache_stats.get('hits', 0)
        misses = cache_stats.get('misses', 0)
        size = cache_stats.get('size', 0)

        # Couleur selon le hit rate
        if hit_rate > 70:
            color = COLORS['secondary']  # Vert
        elif hit_rate > 40:
            color = COLORS['accent']  # Orange
        else:
            color = COLORS['text_muted']  # Gris

        return html.Div([
            html.Div([
                html.Span("📊 Stats du Cache", style={'fontWeight': '600', 'color': COLORS['text'], 'marginRight': '15px'}),
                html.Span(f"Hit Rate: {hit_rate}%", style={'color': color, 'fontWeight': '500', 'marginRight': '15px'}),
                html.Span(f"Hits: {hits}", style={'color': COLORS['text_muted'], 'marginRight': '15px'}),
                html.Span(f"Misses: {misses}", style={'color': COLORS['text_muted'], 'marginRight': '15px'}),
                html.Span(f"Taille: {size}", style={'color': COLORS['text_muted']}),
            ], style={
                'padding': '12px',
                'backgroundColor': COLORS['card_bg'],
                'borderRadius': '8px',
                'border': f'1px solid {COLORS["border"]}'
            })
        ])

    @app.callback(
        Output('rag-status', 'children', allow_duplicate=True),
        [Input('config-cache-enabled', 'value'),
         Input('config-n-context', 'value'),
         Input('config-hybrid-search', 'value'),
         Input('config-reranking', 'value')],
        prevent_initial_call=True
    )
    def update_rag_config(cache_enabled, n_context, hybrid_enabled, rerank_enabled):
        """Applique la configuration au RAG."""
        rag = get_rag_engine()

        # Appliquer les configs
        rag.use_cache = 'enabled' in (cache_enabled or [])
        rag.use_hybrid_search = 'enabled' in (hybrid_enabled or [])
        rag.use_reranking = 'enabled' in (rerank_enabled or [])

        # Retourner un statut (on garde le précédent badge)
        raise PreventUpdate


def register_analytics_callback(app):
    """Callbacks pour les analytics avancées."""

    # Toggle de la collapse
    @app.callback(
        Output('analytics-collapse', 'is_open'),
        [Input('toggle-analytics', 'n_clicks')],
        [State('analytics-collapse', 'is_open')]
    )
    def toggle_analytics(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    # Mise à jour des graphiques analytics
    @app.callback(
        [Output('wordcloud-graph', 'figure'),
         Output('topics-graph', 'figure'),
         Output('network-graph', 'figure'),
         Output('heatmap-graph', 'figure')],
        [Input('stored-data', 'data'),
         Input('sender-dropdown', 'value'),
         Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date')]
    )
    def update_analytics(stored_data, selected_senders, start_date, end_date):
        if not stored_data:
            # Retourner graphiques vides
            empty_fig = apply_dark_theme(px.scatter())
            return empty_fig, empty_fig, empty_fig, empty_fig

        try:
            # Charger et filtrer les données
            df = pd.read_json(stored_data, orient='split')
            df = filter_messages(df, start_date, end_date, selected_senders)

            analytics = get_analytics()

            # 1. Word Cloud (scatter plot avec tailles)
            wordcloud_data = analytics.compute_word_cloud_data(df, top_n=50)

            if wordcloud_data:
                import plotly.graph_objects as go

                # Créer positions aléatoires mais déterministes
                import random
                random.seed(42)  # Pour reproductibilité

                words = [item['word'] for item in wordcloud_data]
                frequencies = [item['frequency'] for item in wordcloud_data]
                sizes = [item['size'] for item in wordcloud_data]

                # Positions aléatoires
                x_pos = [random.uniform(0, 100) for _ in words]
                y_pos = [random.uniform(0, 100) for _ in words]

                wordcloud_fig = go.Figure()
                wordcloud_fig.add_trace(go.Scatter(
                    x=x_pos,
                    y=y_pos,
                    mode='text',
                    text=words,
                    textfont=dict(
                        size=sizes,
                        color=[PLOTLY_COLORS[i % len(PLOTLY_COLORS)] for i in range(len(words))]
                    ),
                    hovertemplate='<b>%{text}</b><br>Fréquence: %{customdata}<extra></extra>',
                    customdata=frequencies
                ))

                wordcloud_fig.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis=dict(visible=False, range=[0, 100]),
                    yaxis=dict(visible=False, range=[0, 100]),
                    hovermode='closest',
                    margin=dict(l=0, r=0, t=0, b=0)
                )
            else:
                wordcloud_fig = apply_dark_theme(px.scatter())
                wordcloud_fig.update_layout(
                    annotations=[{
                        'text': 'Pas assez de données',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': COLORS['text_muted']}
                    }]
                )

            # 2. Topics (barres horizontales)
            topics = analytics.compute_topic_distribution(df, n_topics=5, keywords_per_topic=5)

            if topics:
                import plotly.graph_objects as go

                topics_fig = go.Figure()

                for topic in topics:
                    keywords_str = ', '.join(topic['keywords'][:3])  # Top 3 mots
                    topics_fig.add_trace(go.Bar(
                        y=[f"Topic {topic['topic_id']}: {keywords_str}"],
                        x=[topic['weight']],
                        orientation='h',
                        marker=dict(color=PLOTLY_COLORS[topic['topic_id'] % len(PLOTLY_COLORS)]),
                        hovertemplate=f"<b>Topic {topic['topic_id']}</b><br>" +
                                     f"Mots-clés: {', '.join(topic['keywords'])}<br>" +
                                     f"Poids: {topic['weight']:.2%}<extra></extra>"
                    ))

                topics_fig.update_layout(
                    **PLOTLY_LAYOUT,
                    showlegend=False,
                    xaxis_title="Poids",
                    yaxis=dict(autorange='reversed'),
                    margin=dict(l=200, r=20, t=20, b=40)
                )
            else:
                topics_fig = apply_dark_theme(px.bar())
                topics_fig.update_layout(
                    annotations=[{
                        'text': 'Pas assez de données',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': COLORS['text_muted']}
                    }]
                )

            # 3. Network Graph
            network = analytics.compute_interaction_network(df, time_window_minutes=5)

            if network['nodes'] and network['edges']:
                import plotly.graph_objects as go
                import networkx as nx

                # Créer graphe NetworkX pour layout
                G = nx.Graph()
                for node in network['nodes']:
                    G.add_node(node['id'], **node)
                for edge in network['edges']:
                    G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

                # Layout spring
                pos = nx.spring_layout(G, seed=42)

                # Edges
                edge_x = []
                edge_y = []
                edge_weights = []
                for edge in network['edges']:
                    x0, y0 = pos[edge['source']]
                    x1, y1 = pos[edge['target']]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights.append(edge['weight'])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color=COLORS['text_muted']),
                    hoverinfo='none',
                    mode='lines'
                )

                # Nodes
                node_x = []
                node_y = []
                node_text = []
                node_size = []
                node_hover = []

                for node in network['nodes']:
                    x, y = pos[node['id']]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node['label'])
                    node_size.append(min(node['messages'] * 3, 50))  # Limite taille
                    node_hover.append(f"<b>{node['label']}</b><br>{node['messages']} messages")

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition='top center',
                    marker=dict(
                        size=node_size,
                        color=PLOTLY_COLORS[0],
                        line=dict(width=2, color=COLORS['bg'])
                    ),
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=node_hover,
                    textfont=dict(color=COLORS['text'])
                )

                network_fig = go.Figure(data=[edge_trace, node_trace])
                network_fig.update_layout(
                    **PLOTLY_LAYOUT,
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    hovermode='closest',
                    margin=dict(l=0, r=0, t=0, b=0)
                )
            else:
                network_fig = apply_dark_theme(px.scatter())
                network_fig.update_layout(
                    annotations=[{
                        'text': 'Pas assez d\'interactions détectées',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': COLORS['text_muted']}
                    }]
                )

            # 4. Activity Heatmap
            heatmap_data = analytics.compute_activity_heatmap(df)

            if heatmap_data['data']:
                import plotly.graph_objects as go

                heatmap_fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data['data'],
                    x=heatmap_data['hours'],
                    y=heatmap_data['days'],
                    colorscale='Viridis',
                    hovertemplate='%{y}<br>%{x}h: %{z} messages<extra></extra>',
                    colorbar=dict(
                        title='Messages',
                        tickfont=dict(color=COLORS['text'])
                    )
                ))

                heatmap_fig.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title='Heure',
                    yaxis_title='Jour',
                    xaxis=dict(
                        tickmode='linear',
                        tick0=0,
                        dtick=2,
                        gridcolor=COLORS['border']
                    ),
                    yaxis=dict(gridcolor=COLORS['border'])
                )
            else:
                heatmap_fig = apply_dark_theme(px.imshow([[0]]))
                heatmap_fig.update_layout(
                    annotations=[{
                        'text': 'Pas assez de données temporelles',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': COLORS['text_muted']}
                    }]
                )

            return wordcloud_fig, topics_fig, network_fig, heatmap_fig

        except Exception as e:
            print(f"❌ Erreur analytics: {str(e)}")
            empty_fig = apply_dark_theme(px.scatter())
            return empty_fig, empty_fig, empty_fig, empty_fig


def register_export_callback(app):
    """Callback pour l'export des données."""

    @app.callback(
        Output('download-data', 'data'),
        [Input('export-button', 'n_clicks')],
        [State('stored-data', 'data')]
    )
    def export_data(n_clicks, stored_data):
        if not n_clicks or not stored_data:
            raise PreventUpdate

        try:
            messages = pd.read_json(stored_data, orient='split')
            return dcc.send_data_frame(messages.to_csv, "exported_messages.csv")
        except Exception as e:
            # En cas d'erreur, on ne peut pas afficher de message dans ce callback
            # car il retourne seulement des données de téléchargement
            # L'erreur sera visible dans la console
            print(f"❌ Erreur lors de l'export: {str(e)}")
            raise PreventUpdate
