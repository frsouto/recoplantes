import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import base64

# Charger le dataframe
df = pd.read_csv('../../reports/metriques2.csv')

# Ajouter une colonne d'identifiant unique
df['id'] = df.index

# Créer l'application Dash avec suppression des exceptions de callback
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Liste des métriques à afficher
metrics = ['sharpness', 'sobel', 'brightness', 'noise', 'snr', 'contrast', 'entropy', 'saturation', 'ratiozoom', 'densite','pourcentage_vert']

# Fonction pour créer un boxen plot pour une métrique donnée groupé par plante_maladie
def create_boxen_plot(metric):
    fig = px.box(df, x='plante_maladie', y=metric, points="all", title=f'Distribution de la {metric.capitalize()} par Plante_Maladie')
    fig.update_layout(width=1200, height=1200, showlegend=False)
    return fig

# Définir la mise en page de l'application avec des onglets
app.layout = html.Div([
    dcc.Tabs(id='tabs-metrics', value='sharpness', children=[
        dcc.Tab(label=metric.capitalize(), value=metric) for metric in metrics
    ]),
    dcc.Store(id='selected-metric', data='sharpness'),
    html.Div(id='content', style={'display': 'flex'}),
])

# Définir le callback pour mettre à jour le contenu en fonction de l'onglet sélectionné
@app.callback(
    [Output('content', 'children'),
     Output('selected-metric', 'data')],
    [Input('tabs-metrics', 'value')]
)
def render_content(tab):
    fig = create_boxen_plot(tab)
    return [
        dcc.Graph(id='boxenplot', figure=fig, style={'flex': '1'}),
        html.Div(id='image-container', style={'flex': '1', 'textAlign': 'center', 'padding': '20px'})
    ], tab

# Définir le callback pour mettre à jour l'image en fonction du point cliqué
@app.callback(
    Output('image-container', 'children'),
    [Input('boxenplot', 'clickData')],
    [Input('selected-metric', 'data')]
)
def display_image(clickData, metric):
    if clickData is None:
        return html.P("Cliquez sur un point pour voir l'image correspondante.")
    
    # Extraire les données du point cliqué
    selected_point = clickData['points'][0]
    point_id = selected_point['pointIndex']  # Récupérer l'identifiant unique

    # Filtrer le DataFrame pour récupérer les informations de l'image correspondante
    image_info = df.iloc[point_id]

    # Extraire le chemin de l'image
    image_path = image_info['image_path']
    
    # Lire l'image et la convertir en base64
    encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
    
    return html.Div([
        html.P(f"Chemin de l'image : {image_path}"),
        html.Img(src=f'data:image/jpeg;base64,{encoded_image}', style={'width': '256px', 'height': '256px'})
    ])

# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)