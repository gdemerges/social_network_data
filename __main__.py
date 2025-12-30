"""
Social Network Data Analyzer - Application principale.
Dashboard pour analyser les messages avec RAG et LLM local.
"""

import os
import dash
import dash_bootstrap_components as dbc

from dashboard import create_layout, register_callbacks


def create_app() -> dash.Dash:
    """Crée et configure l'application Dash."""
    application = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Analyseur de Messages"
    )

    application.layout = create_layout()
    register_callbacks(application)

    return application


app = create_app()


if __name__ == '__main__':
    # Utiliser variable d'environnement pour le mode debug (par défaut: False)
    debug_mode = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')
    port = int(os.getenv('PORT', '8050'))
    host = os.getenv('HOST', '127.0.0.1')

    app.run(debug=debug_mode, host=host, port=port)
