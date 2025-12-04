"""
Styles CSS personnalisés pour le dashboard.
"""

# Couleurs du thème
COLORS = {
    'primary': '#6366f1',      # Indigo
    'secondary': '#10b981',    # Emerald (vert pour succès)
    'accent': '#f59e0b',       # Amber (pour avertissements)
    'purple': '#8b5cf6',       # Violet
    'success': '#10b981',      # Emerald
    'warning': '#f59e0b',      # Amber
    'danger': '#ef4444',       # Red
    'dark': '#1e1b4b',         # Indigo très foncé
    'light': '#f8fafc',        # Slate très clair
    'background': '#0f172a',   # Slate 900
    'card_bg': '#1e293b',      # Slate 800
    'card_hover': '#334155',   # Slate 700 (hover)
    'text': '#e2e8f0',         # Slate 200
    'text_muted': '#94a3b8',   # Slate 400
    'border': '#334155',       # Slate 700
    'gradient_start': '#6366f1',
    'gradient_end': '#8b5cf6',
}

# Style de la page principale
PAGE_STYLE = {
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'padding': '20px',
}

# Style du header
HEADER_STYLE = {
    'background': f'linear-gradient(135deg, {COLORS["gradient_start"]} 0%, {COLORS["gradient_end"]} 100%)',
    'padding': '30px',
    'borderRadius': '20px',
    'marginBottom': '30px',
    'boxShadow': '0 10px 40px rgba(99, 102, 241, 0.3)',
}

HEADER_TITLE_STYLE = {
    'color': 'white',
    'fontWeight': '700',
    'fontSize': '2.5rem',
    'marginBottom': '10px',
    'textShadow': '0 2px 4px rgba(0,0,0,0.2)',
}

HEADER_SUBTITLE_STYLE = {
    'color': 'rgba(255,255,255,0.8)',
    'fontSize': '1.1rem',
    'marginBottom': '0',
}

# Style des cartes
CARD_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'border': f'1px solid {COLORS["border"]}',
    'borderRadius': '16px',
    'padding': '24px',
    'marginBottom': '20px',
    'boxShadow': '0 4px 20px rgba(0,0,0,0.2)',
}

CARD_HEADER_STYLE = {
    'color': COLORS['text'],
    'fontWeight': '600',
    'fontSize': '1.2rem',
    'marginBottom': '20px',
    'display': 'flex',
    'alignItems': 'center',
    'gap': '10px',
}

# Style de la zone d'upload
UPLOAD_ZONE_STYLE = {
    'width': '100%',
    'padding': '40px',
    'border': f'2px dashed {COLORS["border"]}',
    'borderRadius': '16px',
    'backgroundColor': COLORS['card_bg'],
    'textAlign': 'center',
    'cursor': 'pointer',
    'transition': 'all 0.3s ease',
}

UPLOAD_ICON_STYLE = {
    'fontSize': '3rem',
    'marginBottom': '15px',
    'color': COLORS['primary'],
}

UPLOAD_TEXT_STYLE = {
    'color': COLORS['text'],
    'fontSize': '1.1rem',
    'marginBottom': '5px',
}

UPLOAD_SUBTEXT_STYLE = {
    'color': COLORS['text_muted'],
    'fontSize': '0.9rem',
}

# Style des graphiques
GRAPH_CARD_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'border': f'1px solid {COLORS["border"]}',
    'borderRadius': '16px',
    'padding': '20px',
    'height': '100%',
    'boxShadow': '0 4px 20px rgba(0,0,0,0.2)',
}

# Style du chat
CHAT_CONTAINER_STYLE = {
    'height': '450px',
    'overflowY': 'auto',
    'backgroundColor': '#0f172a',
    'borderRadius': '16px',
    'padding': '20px',
    'marginBottom': '20px',
    'border': f'1px solid {COLORS["border"]}',
}

CHAT_MESSAGE_USER_STYLE = {
    'backgroundColor': COLORS['primary'],
    'color': 'white',
    'padding': '12px 18px',
    'borderRadius': '18px 18px 4px 18px',
    'marginBottom': '12px',
    'marginLeft': '25%',
    'maxWidth': '75%',
    'boxShadow': '0 2px 8px rgba(99, 102, 241, 0.3)',
}

CHAT_MESSAGE_ASSISTANT_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'color': COLORS['text'],
    'padding': '12px 18px',
    'borderRadius': '18px 18px 18px 4px',
    'marginBottom': '12px',
    'marginRight': '25%',
    'maxWidth': '75%',
    'border': f'1px solid {COLORS["border"]}',
}

CHAT_INPUT_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'border': f'1px solid {COLORS["border"]}',
    'borderRadius': '25px 0 0 25px',
    'color': COLORS['text'],
    'padding': '12px 20px',
    'fontSize': '1rem',
}

CHAT_BUTTON_STYLE = {
    'background': f'linear-gradient(135deg, {COLORS["gradient_start"]} 0%, {COLORS["gradient_end"]} 100%)',
    'border': 'none',
    'borderRadius': '0 25px 25px 0',
    'padding': '12px 25px',
    'fontWeight': '600',
}

# Style des filtres
FILTER_DROPDOWN_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'border': f'1px solid {COLORS["border"]}',
    'borderRadius': '10px',
    'color': COLORS['text'],
}

# Style des badges de statut
STATUS_BADGE_ONLINE = {
    'backgroundColor': COLORS['success'],
    'color': 'white',
    'padding': '6px 12px',
    'borderRadius': '20px',
    'fontSize': '0.85rem',
    'fontWeight': '500',
}

STATUS_BADGE_OFFLINE = {
    'backgroundColor': COLORS['danger'],
    'color': 'white',
    'padding': '6px 12px',
    'borderRadius': '20px',
    'fontSize': '0.85rem',
    'fontWeight': '500',
}

# Style des statistiques
STAT_CARD_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'border': f'1px solid {COLORS["border"]}',
    'borderRadius': '12px',
    'padding': '20px',
    'textAlign': 'center',
}

STAT_VALUE_STYLE = {
    'fontSize': '2rem',
    'fontWeight': '700',
    'color': COLORS['primary'],
    'marginBottom': '5px',
}

STAT_LABEL_STYLE = {
    'fontSize': '0.9rem',
    'color': COLORS['text_muted'],
}

# Style du bouton d'export
EXPORT_BUTTON_STYLE = {
    'background': f'linear-gradient(135deg, {COLORS["success"]} 0%, #059669 100%)',
    'border': 'none',
    'borderRadius': '12px',
    'padding': '12px 30px',
    'fontSize': '1rem',
    'fontWeight': '600',
    'boxShadow': '0 4px 15px rgba(16, 185, 129, 0.3)',
    'transition': 'all 0.3s ease',
}

# Configuration des graphiques Plotly (thème sombre)
PLOTLY_LAYOUT = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': COLORS['text']},
    'xaxis': {
        'gridcolor': COLORS['border'],
        'linecolor': COLORS['border'],
    },
    'yaxis': {
        'gridcolor': COLORS['border'],
        'linecolor': COLORS['border'],
    },
    'margin': {'l': 40, 'r': 20, 't': 40, 'b': 40},
}

PLOTLY_COLORS = [
    COLORS['primary'],
    COLORS['secondary'],
    COLORS['success'],
    COLORS['warning'],
    '#ec4899',  # Pink
    '#14b8a6',  # Teal
]
