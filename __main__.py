"""
Social Network Data Analyzer - Application principale.
Dashboard pour analyser les messages avec RAG et LLM local.
"""

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
    app.run(debug=True)
