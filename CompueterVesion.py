import tkinter as tk
from tkinter import Menu, filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traitement d'image")
        
        # Variables
        self.original_image = None
        self.processed_image = None
        
        # Menu
        menu = Menu(root)
        root.config(menu=menu)
        
        file_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Ouvrir", command=self.open_image)
        file_menu.add_command(label="Enregistrer", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=root.quit)
        
        filter_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Filtres", menu=filter_menu)
        
        # Passe-bas
        low_pass_menu = Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Filtrage Passe Bas", menu=low_pass_menu)
        low_pass_menu.add_command(label="Moyenneur(3x3)", command=lambda: self.apply_filter("mean", 3))
        low_pass_menu.add_command(label="Moyenneur(5x5)", command=lambda: self.apply_filter("mean", 5))
        low_pass_menu.add_command(label="Gaussien(3x3)", command=lambda: self.apply_filter("gaussian", 3))
        low_pass_menu.add_command(label="Gaussien(5x5)", command=lambda: self.apply_filter("gaussian", 5))
        
        # Passe-haut
        high_pass_menu = Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Filtrage Passe Haut", menu=high_pass_menu)
        high_pass_menu.add_command(label="Conique", command=lambda: self.apply_filter("conic"))
        high_pass_menu.add_command(label="Pyramidal", command=lambda: self.apply_filter("pyramidal"))
        high_pass_menu.add_separator()
        high_pass_menu.add_command(label="Gradient", command=lambda: self.apply_filter("gradient"))
        high_pass_menu.add_command(label="Sobel", command=lambda: self.apply_filter("sobel"))
        high_pass_menu.add_command(label="Prewitt", command=lambda: self.apply_filter("prewitt"))
        high_pass_menu.add_command(label="Roberts", command=lambda: self.apply_filter("roberts"))
        high_pass_menu.add_command(label="Laplacien", command=lambda: self.apply_filter("laplacian"))
        
        # Frames pour afficher les images
        self.original_frame = ttk.LabelFrame(root, text="Image originale")
        self.original_frame.grid(row=0, column=0, padx=10, pady=10)
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack()
        
        self.processed_frame = ttk.LabelFrame(root, text="Image traitée")
        self.processed_frame.grid(row=0, column=1, padx=10, pady=10)
        self.processed_label = ttk.Label(self.processed_frame)
        self.processed_label.pack()
    
    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.bmp")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.original_image, self.original_label)
    
    def save_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))
        else:
            tk.messagebox.showerror("Erreur", "Aucune image traitée à enregistrer.")
    
    def display_image(self, image, label):
        # Limiter la taille de l'image (400x400 pixels maximum)
        max_size = 400
        h, w = image.shape[:2]
        scale = min(max_size / h, max_size / w)
        new_size = (int(w * scale), int(h * scale))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        # Convertir l'image pour tkinter
        image = Image.fromarray(resized_image)
        image = ImageTk.PhotoImage(image)
        label.config(image=image)
        label.image = image
    
    def apply_filter(self, filter_type, size=None):
        if self.original_image is None:
            tk.messagebox.showerror("Erreur", "Veuillez charger une image d'abord.")
            return
        
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        
        if filter_type == "mean":
            kernel = np.ones((size, size), np.float32) / (size * size)
            self.processed_image = cv2.filter2D(self.original_image, -1, kernel)
        elif filter_type == "gaussian":
            self.processed_image = cv2.GaussianBlur(self.original_image, (size, size), 0)
        elif filter_type == "conic":
            kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
            self.processed_image = cv2.filter2D(self.original_image, -1, kernel)
        elif filter_type == "pyramidal":
            self.processed_image = cv2.pyrUp(cv2.pyrDown(self.original_image))
        elif filter_type == "gradient":
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient = cv2.magnitude(sobelx, sobely)
            self.processed_image = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif filter_type == "sobel":
            self.processed_image = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
            self.processed_image = cv2.convertScaleAbs(self.processed_image)
        elif filter_type == "prewitt":
            kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            prewittx = cv2.filter2D(gray_image, -1, kernelx)
            prewitty = cv2.filter2D(gray_image, -1, kernely)
            self.processed_image = cv2.add(prewittx, prewitty)
        elif filter_type == "roberts":
            kernelx = np.array([[1, 0], [0, -1]])
            kernely = np.array([[0, 1], [-1, 0]])
            robertsx = cv2.filter2D(gray_image, -1, kernelx)
            robertsy = cv2.filter2D(gray_image, -1, kernely)
            self.processed_image = cv2.add(robertsx, robertsy)
        elif filter_type == "laplacian":
            self.processed_image = cv2.Laplacian(gray_image, cv2.CV_64F)
            self.processed_image = cv2.convertScaleAbs(self.processed_image)
        
        self.display_image(self.processed_image, self.processed_label)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
