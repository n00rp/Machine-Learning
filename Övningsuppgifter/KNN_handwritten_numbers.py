import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps, ImageEnhance, ImageGrab
import numpy as np
from tensorflow import keras

# Ladda den tränade modellen
model = joblib.load('Övningsuppgifter/knn_model.pkl')

# Funktion för att klassificera en uppladdad bild
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Öppna och förbehandla bilden
            image = Image.open(file_path).convert('L')
            image = ImageOps.invert(image)  # Invertera färger
            image = ImageEnhance.Contrast(image).enhance(2.0)  # Öka kontrasten
            image = image.resize((28, 28))
            image_array = np.array(image)
            
            # Normalisera pixelvärden till 0-1
            image_array = image_array / 255.0
            
            # Ändra form till 1D-vektor
            image_array = image_array.reshape(1, -1)
            
            # Gör förutsägelse
            prediction = model.predict(image_array)
            print(f'Förutsagd siffra: {prediction[0]}')
            result_label.config(text=f'Förutsagd siffra: {prediction[0]}')
        except Exception as e:
            messagebox.showerror('Fel', f'Kunde inte klassificera bilden: {str(e)}')

# Funktion för att klassificera den ritade bilden
def classify_drawing():
    try:
        # Spara det ritade området som en bild
        x = root.winfo_rootx() + canvas.winfo_x()
        y = root.winfo_rooty() + canvas.winfo_y()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()
        image = ImageGrab.grab().crop((x, y, x1, y1))
        
        # Förbehandla bilden
        image = image.convert('L')
        image = ImageOps.invert(image)
        image = ImageEnhance.Contrast(image).enhance(2.0)
        image = image.resize((28, 28))
        image_array = np.array(image)
        
        # Normalisera pixelvärden till 0-1
        image_array = image_array / 255.0
        
        # Ändra form till 1D-vektor
        image_array = image_array.reshape(1, -1)
        
        # Gör förutsägelse
        prediction = model.predict(image_array)
        print(f'Förutsagd siffra: {prediction[0]}')
        result_label.config(text=f'Förutsagd siffra: {prediction[0]}')
    except Exception as e:
        result_label.config(text=f'Fel: {str(e)}')

# Funktion för att rensa ritområdet
def clear_canvas():
    canvas.delete('all')
    result_label.config(text='')

# Funktion för att rita på Canvas
def start_drawing(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw(event):
    global last_x, last_y
    canvas.create_line((last_x, last_y, event.x, event.y), fill='black', width=10)
    last_x, last_y = event.x, event.y

# Funktion för att testa modellen med kända bilder från MNIST
def test_model_with_mnist():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    for i in range(10):  # Testa de första 10 bilderna
        image = X_test[i]
        image = Image.fromarray(image)  # Konvertera till PIL-bild
        image = image.convert('L')
        image = ImageOps.invert(image)
        image = ImageEnhance.Contrast(image).enhance(2.0)
        image = image.resize((28, 28))
        image_array = np.array(image)
        
        # Normalisera pixelvärden till 0-1
        image_array = image_array / 255.0
        
        # Ändra form till 1D-vektor
        image_array = image_array.reshape(1, -1)
        
        # Gör förutsägelse
        prediction = model.predict(image_array)
        print(f'Förutsagd siffra för bild {i}: {prediction[0]} (Verklig siffra: {y_test[i]})')

# Anropa testfunktionen
test_model_with_mnist()  

# Skapa huvudfönstret
root = tk.Tk()
root.title('Klassificera handskrivna siffror')

# Knapp för att ladda upp en bild
upload_button = tk.Button(root, text='Ladda upp bild', command=classify_image)
upload_button.pack(pady=20)

# Skapa ett ritområde
canvas = tk.Canvas(root, width=280, height=280, bg='white')
canvas.pack(pady=20)

# Binda musdrag till ritfunktioner
canvas.bind('<Button-1>', start_drawing)
canvas.bind('<B1-Motion>', draw)

# Knapp för att klassificera den ritade bilden
classify_button = tk.Button(root, text='Klassificera', command=classify_drawing)
classify_button.pack(pady=10)

# Knapp för att rensa ritområdet
clear_button = tk.Button(root, text='Rensa', command=clear_canvas)
clear_button.pack(pady=10)

# Label för att visa resultatet
result_label = tk.Label(root, text='', font=('Arial', 18))
result_label.pack(pady=20)

# Starta huvudloopen
root.mainloop()