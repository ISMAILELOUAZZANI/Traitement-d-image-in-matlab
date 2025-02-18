import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


def apply_morphology():
    global img, processed_img
    if img is None:
        return
    
    operation = operation_var.get()
    threshold = int(threshold_var.get())
    kernel = np.ones((5,5), np.uint8)
    
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    if operation == "Dilation":
        processed_img = cv2.dilate(binary, kernel, iterations=1)
    elif operation == "Erosion":
        processed_img = cv2.erode(binary, kernel, iterations=1)
    elif operation == "Fermuture":
        processed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    elif operation == "Ouverture":
        processed_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif operation == "Top Hat Dark":
        processed_img = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
    elif operation == "Gradient":
        processed_img = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    else:
        processed_img = binary
    
    show_images()

def show_images():
    global img, processed_img
    if img is None or processed_img is None:
        return
    
    img_resized = cv2.resize(img, (250, 250))
    proc_resized = cv2.resize(processed_img, (250, 250))
    
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_resized))
    proc_tk = ImageTk.PhotoImage(image=Image.fromarray(proc_resized))
    
    label_original.config(image=img_tk)
    label_original.image = img_tk
    
    label_processed.config(image=proc_tk)
    label_processed.image = proc_tk

def load_image():
    global img
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        show_images()

# GUI Setup
root = tk.Tk()
root.title("Morphological Operators")

tk.Button(root, text="Charger l'mage", command=load_image).pack()

operation_var = tk.StringVar()
operation_var.set("Dilation")
ttk.Combobox(root, textvariable=operation_var, values=["Dilation", "Erosion", "Fermuture", "Ouverture", "Top Hat Dark", "Gradient"]).pack()

threshold_var = tk.StringVar()
threshold_var.set("127")
tk.Entry(root, textvariable=threshold_var).pack()

tk.Button(root, text="Run", command=apply_morphology).pack()

label_original = tk.Label(root)
label_original.pack()

label_processed = tk.Label(root)
label_processed.pack()

img = None
processed_img = None

root.mainloop()
