import tkinter as tk
from tkinter import Menu, filedialog, ttk, messagebox
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

        # Menu Fichier
        file_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Ouvrir", command=self.open_image)
        file_menu.add_command(label="Enregistrer", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=root.quit)

        # Menu Transformation
        transformation_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Transformation", menu=transformation_menu)
        transformation_menu.add_command(label="Inversion des couleurs", command=self.invert_colors)
        transformation_menu.add_command(label="Contraste", command=self.adjust_contrast)
        transformation_menu.add_command(label="Luminosité", command=self.adjust_brightness)
        transformation_menu.add_command(label="Binarisation", command=self.binarize_image)
        transformation_menu.add_command(label="Niveau de gris", command=self.convert_to_grayscale)
        transformation_menu.add_command(label="Histogramme", command=self.display_histogram)

        # Menu Filtrage
        filter_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Filtrage", menu=filter_menu)

        # Passe-bas
        low_pass_menu = Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Passe-bas", menu=low_pass_menu)
        low_pass_menu.add_command(label="Moyenneur (3x3)", command=lambda: self.apply_filter("mean", 3))
        low_pass_menu.add_command(label="Moyenneur (5x5)", command=lambda: self.apply_filter("mean", 5))
        low_pass_menu.add_command(label="Gaussien (3x3)", command=lambda: self.apply_filter("gaussian", 3))
        low_pass_menu.add_command(label="Gaussien (5x5)", command=lambda: self.apply_filter("gaussian", 5))

        # Passe-haut
        high_pass_menu = Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Passe-haut", menu=high_pass_menu)
        high_pass_menu.add_command(label="Sobel", command=lambda: self.apply_filter("sobel"))
        high_pass_menu.add_command(label="Prewitt", command=lambda: self.apply_filter("prewitt"))
        high_pass_menu.add_command(label="Roberts", command=lambda: self.apply_filter("roberts"))
        high_pass_menu.add_command(label="Laplacien", command=lambda: self.apply_filter("laplacian"))

        # Menu Bruit
        noise_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Bruit", menu=noise_menu)
        noise_menu.add_command(label="Gaussien", command=self.add_gaussian_noise)
        noise_menu.add_command(label="Poivre et sel", command=self.add_salt_and_pepper_noise)

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
            messagebox.showerror("Erreur", "Aucune image traitée à enregistrer.")

    def display_image(self, image, label):
        max_size = 400
        h, w = image.shape[:2]
        scale = min(max_size / h, max_size / w)
        new_size = (int(w * scale), int(h * scale))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        image = Image.fromarray(resized_image)
        image = ImageTk.PhotoImage(image)
        label.config(image=image)
        label.image = image

    def invert_colors(self):
        if self.original_image is not None:
            self.processed_image = cv2.bitwise_not(self.original_image)
            self.display_image(self.processed_image, self.processed_label)

    def adjust_contrast(self):
        if self.original_image is not None:
            alpha = 1.5
            self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=0)
            self.display_image(self.processed_image, self.processed_label)

    def adjust_brightness(self):
        if self.original_image is not None:
            beta = 50
            self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=1, beta=beta)
            self.display_image(self.processed_image, self.processed_label)

    def binarize_image(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            _, self.processed_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
            self.display_image(self.processed_image, self.processed_label)

    def convert_to_grayscale(self):
        if self.original_image is not None:
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            self.display_image(self.processed_image, self.processed_label)

    def display_histogram(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            self.processed_image = np.zeros((300, 256, 3), dtype=np.uint8)
            hist_max = np.max(histogram)
            for x, y in enumerate(histogram):
                cv2.line(self.processed_image, (x, 300), (x, 300 - int((y / hist_max) * 300)), (255, 255, 255))
            self.display_image(self.processed_image, self.processed_label)

    def add_gaussian_noise(self):
        if self.original_image is not None:
            row, col, ch = self.original_image.shape
            mean = 0
            sigma = 15
            gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.uint8)
            self.processed_image = cv2.add(self.original_image, gauss)
            self.display_image(self.processed_image, self.processed_label)

    def add_salt_and_pepper_noise(self):
        if self.original_image is not None:
            s_vs_p = 0.5
            amount = 0.02
            noisy_image = self.original_image.copy()
            num_salt = np.ceil(amount * self.original_image.size * s_vs_p).astype(int)
            num_pepper = np.ceil(amount * self.original_image.size * (1.0 - s_vs_p)).astype(int)

            coords = [np.random.randint(0, i - 1, num_salt) for i in self.original_image.shape]
            noisy_image[coords[0], coords[1], :] = 255

            coords = [np.random.randint(0, i - 1, num_pepper) for i in self.original_image.shape]
            noisy_image[coords[0], coords[1], :] = 0

            self.processed_image = noisy_image
            self.display_image(self.processed_image, self.processed_label)

    def apply_filter(self, filter_type, kernel_size=3):
        if self.original_image is None:
            messagebox.showerror("Erreur", "Veuillez charger une image d'abord.")
            return

        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        if filter_type == "mean":
            self.processed_image = cv2.blur(self.original_image, (kernel_size, kernel_size))
        elif filter_type == "gaussian":
            self.processed_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)
        elif filter_type == "sobel":
            self.processed_image = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=kernel_size)
        elif filter_type == "laplacian":
            self.processed_image = cv2.Laplacian(gray_image, cv2.CV_64F)
        elif filter_type == "prewitt":
            kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            self.processed_image = cv2.filter2D(gray_image, -1, kernelx) + cv2.filter2D(gray_image, -1, kernely)
        elif filter_type == "roberts":
            kernelx = np.array([[1, 0], [0, -1]])
            kernely = np.array([[0, 1], [-1, 0]])
            self.processed_image = cv2.filter2D(gray_image, -1, kernelx) + cv2.filter2D(gray_image, -1, kernely)

        self.display_image(self.processed_image, self.processed_label)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
