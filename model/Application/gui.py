import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from model.Application.SimilarImages import get_similar_images

alg = "kmeans"
file_path_global = None
error_label = None
results = [None, None, None, None, None]
thread = None
type_art = None
accuracy_art = None
output_labels = []
input_image_label = None
root = None
dataset = "..\\..\\dataset\\01.mixed\\"
csv_filepath = "../../dataset/WikiArt.csv"
stock_img_tk = None

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
    global results
    similar_images = None
    if file_path_global is None:
        error_label = ttk.Label(root, text = "Caricare prima un'immagine")
        error_label.grid(row = 0, column = 4, padx=10, pady=10)
    else:
        if error_label is not None:
            error_label.destroy()
        similar_images, results = get_similar_images(file_path_global, alg, 5, csv_filepath)

    if similar_images is not None:
        h = 0
        for img_name in similar_images:
            img_path = dataset + str(img_name[0])
            img = Image.open(img_path).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            output_labels[h].configure(image=img_tk)
            output_labels[h].image = img_tk
            h += 1

    if results[3] is not None:
        global type_art
        type_art.destroy()
        # Etichetta per mostrare il tipo di opera d'arte
        type_art = ttk.Label(root, text="Tipo di opera: " + str(results[3]))
        type_art.grid(row=2, column=0, padx=10, pady=10)

    if results[4] is not None:
        global accuracy_art
        accuracy_art.destroy()
        # Etichetta per mostrare l'accuracy della previsione sul tipo di immagine
        accuracy_art = ttk.Label(root, text="Accuracy Previsione: " + str(results[4]))
        accuracy_art.grid(row=2, column=1, padx=10, pady=10)

def estrai_precisione():
    global results
    popup = tk.Toplevel(root)
    popup.title("Precisione del clustering")
    popup.geometry("300x200")

    label3 = tk.Label(popup, text="Algoritmo: " + str(alg))
    label3.pack(pady=10)

    label1 = tk.Label(popup, text="Silhouette score: "+str(results[0]))
    label1.pack(pady=10)

    label2 = tk.Label(popup, text="Davies-Bouldin score: "+str(results[1]))
    label2.pack(pady=10)

    label4 = tk.Label(popup, text="Tempo di esecuzione: " + str(results[2]))
    label4.pack(pady=10)

    close_button = tk.Button(popup, text="Chiudi", command=popup.destroy)
    close_button.pack(pady=10)


def set_kmeans():
    global alg
    alg = "kmeans"

def set_bottomup():
    global alg
    alg = "bottomup"

def set_hdbscan():
    global alg
    alg = "dbscan"

def start_gui(dataset_path, csv_path):
    global type_art
    global accuracy_art
    global output_labels
    global input_image_label
    global root
    global dataset
    global stock_img_tk
    global csv_filepath
    csv_filepath = str(csv_path)
    dataset = str(dataset_path)

    # Configurazione della finestra principale
    root = tk.Tk()
    root.title("ArtAi")
    root.geometry("1400x400")

    menu_bar = tk.Menu(root)
    menu_algoritmi = tk.Menu(menu_bar, tearoff=0)
    menu_algoritmi.add_command(label="Kmeans", command=set_kmeans)
    menu_algoritmi.add_command(label="Bottomup", command=set_bottomup)
    menu_algoritmi.add_command(label="DBSCAN", command=set_hdbscan)

    menu_valutazione = tk.Menu(menu_bar, tearoff=0)
    menu_valutazione.add_command(label="Precisione", command=estrai_precisione)

    menu_bar.add_cascade(label="Algoritmo", menu=menu_algoritmi)
    menu_bar.add_cascade(label="Valutazione", menu=menu_valutazione)
    root.config(menu=menu_bar)

    # Etichetta e pulsante per l'immagine di input
    ttk.Label(root, text="Carica un'immagine:").grid(row=0, column=0, padx=10, pady=10)
    upload_button = ttk.Button(root, text="Scegli file", command=upload_image)
    upload_button.grid(row=0, column=1, padx=10, pady=10)

    # Etichetta e pulsante per l'output delle immagini
    ttk.Label(root, text="Trova Immagini:").grid(row=0, column=2, padx=10, pady=10)
    upload_button = ttk.Button(root, text="Cerca", command=load_similar_images)
    upload_button.grid(row=0, column=3, padx=10, pady=10)

    # Mostra l'immagine di input
    input_image_label = ttk.Label(root)
    input_image_label.grid(row=1, column=0, columnspan=2, pady=20)

    # Etichette per mostrare le immagini simili
    stock_path = "line-drawing-of-an-empty-square-frame-on-a-white_534611_wh860.png"
    stock_img = Image.open(stock_path).resize((200, 200))
    stock_img_tk = ImageTk.PhotoImage(stock_img)

    for i in range(5):
        label = ttk.Label(root)
        label.grid(row=1, column=i + 2, padx=10, pady=10)
        label.configure(image=stock_img_tk)
        label.image = stock_img_tk
        output_labels.append(label)

    type_art = ttk.Label(root)
    accuracy_art = ttk.Label(root)

    # Avvio dell'interfaccia grafica
    root.mainloop()

start_gui(dataset, csv_filepath)