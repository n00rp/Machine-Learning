import dash
from dash import dcc, html, Input, Output, State
from dash_canvas import DashCanvas
import dash_bootstrap_components as dbc
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import joblib
import logging
import time

# Konfigurera logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    filename='app_debug.log',
    filemode='w'
)

# Ladda din tränade KNN-modell
best_knn = joblib.load(r'c:\Programering\It högskolan\Maskininlärning\Machine-Leraning\Övningsuppgifter\knn_model.pkl')

# Skapa Dash-app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout för appen
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Handskriftsigenkänning med KNN"),
            html.P("Rita en siffra i rutan nedan och klicka på 'Klassificera'")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            DashCanvas(
                id='canvas',
                width=280,
                height=280,
                lineWidth=10,
                goButtonTitle='Klassificera'
            ),
            dbc.Button('Klassificera', id='classify-button', n_clicks=0, className="mt-3"),
            dcc.Loading(
                id="loading",
                type="default",
                children=html.Div(id="loading-output")
            )
        ], width=6),
        dbc.Col([
            html.H3("Resultat:"),
            html.Div(id='prediction-output')
        ], width=6)
    ]),
    html.Div(id='dummy-output', style={'display': 'none'})
])

# Debug callback för att visa canvas-data
@app.callback(
    Output('dummy-output', 'children'),
    Input('canvas', 'image_content'),
    State('canvas', 'json_data')
)
def debug_callback(image_content, json_data):
    print("Canvas callback triggered!")
    logging.debug(f"Canvas data: {image_content}")
    logging.debug(f"JSON data: {json_data}")
    if json_data is not None:
        logging.debug(f"JSON data length: {len(json_data)}")
    if image_content is not None:
        logging.debug(f"Image content length: {len(image_content)}")
    if image_content is None or not image_content.strip() or image_content == 'data:,':
        logging.error("Ogiltig image_content")
        return "Ogiltig bilddata"
    
    if json_data is None or not json_data.strip() or json_data == '{}':
        logging.error("Ogiltig json_data")
        return "Ogiltig json_data"
    
    # Kontrollera om bilddata är korrekt formaterad
    if ',' not in image_content:
        logging.error("Bilddata saknar kommatecken")
        return "Bilddata saknar kommatecken"
    
    # Extrahera och validera base64-delen
    try:
        image_data = image_content.split(',')[1]
        base64.b64decode(image_data)
        logging.debug("Bilddata är korrekt formaterad")
        return "Bilddata är korrekt formaterad"
    except Exception as e:
        logging.error(f"Ogiltig base64-data: {str(e)}")
        return f"Ogiltig base64-data: {str(e)}"

# Callback för att hantera klassificering
@app.callback(
    Output('prediction-output', 'children'),
    Output('loading-output', 'children'),
    Input('classify-button', 'n_clicks'),
    State('canvas', 'image_content'),
    State('canvas', 'json_data'),
    prevent_initial_call=True
)
def classify_digit(n_clicks, image_content, json_data):
    print("Classify callback triggered!")
    logging.debug(f"JSON data: {json_data}")
    if n_clicks == 0:
        logging.debug("Ingen bild mottagen")
        return "", ""
    
    if image_content is None or not image_content.strip() or image_content == 'data:,':
        logging.error("Ogiltig image_content")
        return "Du måste rita en siffra innan du klickar på 'Klassificera'.", ""
    
    if json_data is None or not json_data.strip() or json_data == '{}':
        logging.error("Ogiltig json_data")
        return "Du måste rita en siffra innan du klickar på 'Klassificera'.", ""
    
    try:
        logging.debug(f"Bilddata mottagen: {image_content[:100]}...")
        
        # Extrahera bilddata från base64-strängen
        if not image_content or ',' not in image_content:
            logging.error("Ogiltig image_content")
            return "Ogiltig bilddata", ""
        image_data = image_content.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Spara den mottagna bilden
        with open('received_image.png', 'wb') as f:
            f.write(image_bytes)
        logging.debug("Bilden sparad som received_image.png")
        
        # Konvertera bilden till 28x28 grayscale och centrera den
        image = Image.open(BytesIO(image_bytes))
        image = image.resize((28, 28)).convert('L')
        image.save('processed_image.png')
        logging.debug("Bilden bearbetad och sparad som processed_image.png")
        
        # Centrera bilden
        image_array = np.array(image)
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        image_array = image_array.reshape(1, -1)
        logging.debug(f"Bildarray: {image_array}")
        
        # Kontrollera om bildarrayen är korrekt
        if image_array.shape != (1, 784):
            logging.error(f"Felaktig bildarray shape: {image_array.shape}")
            return "Felaktig bildarray shape", ""
        
        # Gör förutsägelse med KNN-modellen
        prediction = best_knn.predict(image_array)
        logging.debug(f"Förutsägelse: {prediction[0]}")
        return f"Modellen tror att du ritade siffran: {prediction[0]}", ""
    except Exception as e:
        logging.error(f"Fel vid klassificering: {str(e)}", exc_info=True)
        return f"Fel vid klassificering: {str(e)}", ""

# Kör appen
if __name__ == '__main__':
    app.run_server(debug=True)