import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from model.Gui.SimilarImages import get_similar_images

alg = "kmeans"
file_path_global = None
error_label = None

def upload_image():
    global file_path_global
    global stock_img_tk
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if not file_path:
        return

    #Reset degli output labels
    for h in range(5):
        output_labels[h].configure(image=stock_img_tk)
        output_labels[h].image = stock_img_tk

    #Mostra l'immagine caricata
    img = Image.open(file_path).resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    input_image_label.configure(image=img_tk)
    input_image_label.image = img_tk

    file_path_global = file_path

def load_similar_images():
    global file_path_global
    global error_label
    similar_images = None
    if file_path_global is None:
        error_label = ttk.Label(root, text = "Caricare prima un'immagine")
        error_label.grid(row = 0, column = 4, padx=10, pady=10)
    else:
        if error_label is not None:
            error_label.destroy()
        similar_images = get_similar_images(file_path_global, alg)

    if similar_images is not None:
        h = 0
        for img_name in similar_images:
            img_path = "F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\dataset\\dataset2\\01.mixed\\" + str(img_name)
            img = Image.open(img_path).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            output_labels[h].configure(image=img_tk)
            output_labels[h].image = img_tk
            h += 1

def set_kmeans():
    global alg
    alg = "kmeans"

def set_bottomup():
    global alg
    alg = "bottomup"

def set_hdbscan():
    global alg
    alg = "dbscan"

# Configurazione della finestra principale
root = tk.Tk()
root.title("ArtAi")
root.geometry("1400x300")

menu_bar = tk.Menu(root)
menu_opzioni = tk.Menu(menu_bar, tearoff=0)
menu_opzioni.add_command(label="Kmeans", command=set_kmeans)
menu_opzioni.add_command(label="Bottomup", command=set_bottomup)
menu_opzioni.add_command(label="DBSCAN", command=set_hdbscan)

menu_bar.add_cascade(label="Algoritmo", menu=menu_opzioni)

root.config(menu=menu_bar)

# Etichetta e pulsante per l'immagine di input
ttk.Label(root, text="Carica un'immagine:").grid(row=0, column=0, padx=10, pady=10)
upload_button = ttk.Button(root, text="Scegli file", command=upload_image)
upload_button.grid(row=0, column=1, padx=10, pady=10)

#Etichetta e pulsante per l'output delle immagini
ttk.Label(root, text="Trova Immagini:").grid(row=0, column=2, padx=10, pady=10)
upload_button = ttk.Button(root, text="Cerca", command=load_similar_images)
upload_button.grid(row=0, column=3, padx=10, pady=10)

# Mostra l'immagine di input
input_image_label = ttk.Label(root)
input_image_label.grid(row=1, column=0, columnspan=2, pady=20)

# Etichette per mostrare le immagini simili
output_labels = []
stock_path = "line-drawing-of-an-empty-square-frame-on-a-white_534611_wh860.png"
stock_img = Image.open(stock_path).resize((200, 200))
stock_img_tk = ImageTk.PhotoImage(stock_img)

for i in range(5):
    label = ttk.Label(root)
    label.grid(row=1, column=i+2, padx=10, pady=10)
    label.configure(image=stock_img_tk)
    label.image = stock_img_tk
    output_labels.append(label)

# Avvio dell'interfaccia grafica
root.mainloop()