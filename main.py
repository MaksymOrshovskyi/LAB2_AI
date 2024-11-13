import tkinter as tk
from tkinter import filedialog, Label, Button, Scale, Text, Toplevel, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class FeatureExtractionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Feature Extraction Tool")

        # Зберігання зображень та векторів
        self.class_images = {'1': [], '2': [], '8': []}
        self.class_vectors = {'1': [], '2': [], '8': []}
        self.class_stats = {'1': {}, '2': {}, '8': {}}

        # Створення вікон для кожного класу
        self.create_class_windows()

        # Елементи управління
        self.threshold_slider = Scale(master, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold")
        self.threshold_slider.set(128)
        self.threshold_slider.pack()

        self.segments_slider = Scale(master, from_=2, to=10, orient=tk.HORIZONTAL, label="Number of Segments")
        self.segments_slider.set(5)
        self.segments_slider.pack()

        Button(master, text="Upload Class 1 Images", command=lambda: self.upload_images('1')).pack()
        Button(master, text="Upload Class 2 Images", command=lambda: self.upload_images('2')).pack()
        Button(master, text="Upload Class 8 Images", command=lambda: self.upload_images('8')).pack()
        Button(master, text="Upload Unknown Image", command=self.upload_unknown_image).pack()
        Button(master, text="Classify Unknown Image", command=self.classify_image).pack()

        self.vector_text_a = Text(master, height=10)
        self.vector_text_a.pack(expand=True, fill='both')
        self.vector_text_b = Text(master, height=10)
        self.vector_text_b.pack(expand=True, fill='both')
        self.vector_text_c = Text(master, height=10)
        self.vector_text_c.pack(expand=True, fill='both')
        self.unknown_vector_text = Text(master, height=5)
        self.unknown_vector_text.pack(expand=True, fill='both')

        self.stats_text = Text(master, height=10)
        self.stats_text.pack(expand=True, fill='both')

        self.unknown_image_path = None
        self.unknown_vector = []

    def create_class_windows(self):
        self.class_windows = {}
        for class_name in ['1', '2', '8']:
            self.class_windows[class_name] = Toplevel(self.master)
            self.class_windows[class_name].title(f"Class {class_name} Images")
            self.class_windows[class_name].geometry("400x400")

    def upload_images(self, class_name):
        filepaths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if filepaths:
            self.class_images[class_name] = list(filepaths)
            self.process_class_images(class_name)

    def process_class_images(self, class_name):
        vectors = []
        for filepath in self.class_images[class_name]:
            vector = self.process_image(filepath)
            if vector:
                vectors.append(vector)

        self.class_vectors[class_name] = vectors
        self.calculate_min_max_stats(class_name)

        s1_vectors = [self.normalize_s1(v) for v in vectors]
        m1_vectors = [self.normalize_m1(v) for v in vectors]

        formatted_vectors = ""
        for i, vector in enumerate(vectors):
            formatted_vectors += f"Absolute Vector {i+1}: {vector}\n"
            formatted_vectors += f"Оршовський S1 {i+1}: {s1_vectors[i]}\n"
            formatted_vectors += f"Оршовський M1 {i+1}: {m1_vectors[i]}\n"

        if class_name == '1':
            self.vector_text_a.delete(1.0, tk.END)
            self.vector_text_a.insert(tk.END, formatted_vectors)
        elif class_name == '2':
            self.vector_text_b.delete(1.0, tk.END)
            self.vector_text_b.insert(tk.END, formatted_vectors)
        elif class_name == '8':
            self.vector_text_c.delete(1.0, tk.END)
            self.vector_text_c.insert(tk.END, formatted_vectors)

        self.display_class_images(class_name)
        self.display_class_stats(class_name)

    def calculate_min_max_stats(self, class_name):
        s1_vectors = [self.normalize_s1(v) for v in self.class_vectors[class_name]]
        m1_vectors = [self.normalize_m1(v) for v in self.class_vectors[class_name]]

        self.class_stats[class_name]['S1MAX'] = np.max(s1_vectors, axis=0)
        self.class_stats[class_name]['S1MIN'] = np.min(s1_vectors, axis=0)
        self.class_stats[class_name]['M1MAX'] = np.max(m1_vectors, axis=0)
        self.class_stats[class_name]['M1MIN'] = np.min(m1_vectors, axis=0)

    def display_class_images(self, class_name):
        window = self.class_windows[class_name]
        for widget in window.winfo_children():
            widget.destroy()

        for filepath in self.class_images[class_name]:
            img = self.draw_segments(filepath)
            if img is not None:
                tk_image = ImageTk.PhotoImage(img)
                image_label = Label(window, image=tk_image)
                image_label.image = tk_image
                image_label.pack(side=tk.LEFT)

    def display_class_stats(self, class_name):
        stats = self.class_stats[class_name]
        stats_text = f"Class {class_name} Stats:\n"
        stats_text += f"Оршовський S1MAX: {stats['S1MAX']}\n"
        stats_text += f"Оршовський S1MIN: {stats['S1MIN']}\n"
        stats_text += f"Оршовський M1MAX: {stats['M1MAX']}\n"
        stats_text += f"Оршовський M1MIN: {stats['M1MIN']}\n"
        self.stats_text.insert(tk.END, stats_text + "\n")

    def draw_segments(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        segments = self.segments_slider.get()
        height, width = img.shape
        segment_width = width // segments
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(1, segments):
            cv2.line(img_color, (i * segment_width, 0), (i * segment_width, height), (255, 0, 0), 1)
        img_pil = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        img_pil = img_pil.resize((100, 100))
        return img_pil

    def process_image(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        _, thresholded = cv2.threshold(img, self.threshold_slider.get(), 255, cv2.THRESH_BINARY)
        segments = self.segments_slider.get()
        height, width = thresholded.shape
        segment_width = width // segments
        absolute_vector = [np.sum(thresholded[:, i * segment_width:(i + 1) * segment_width] == 0) for i in range(segments)]
        return absolute_vector

    def normalize_s1(self, vector):
        total_sum = sum(vector)
        return [x / total_sum for x in vector] if total_sum > 0 else [0] * len(vector)

    def normalize_m1(self, vector):
        max_val = max(vector)
        return [x / max_val for x in vector] if max_val > 0 else [0] * len(vector)

    def upload_unknown_image(self):
        self.unknown_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if self.unknown_image_path:
            self.unknown_vector = self.process_image(self.unknown_image_path)
            if self.unknown_vector:
                s1_vector = self.normalize_s1(self.unknown_vector)
                m1_vector = self.normalize_m1(self.unknown_vector)

                self.unknown_vector_text.delete(1.0, tk.END)
                self.unknown_vector_text.insert(tk.END, f"Path: {self.unknown_image_path}\n")
                self.unknown_vector_text.insert(tk.END, f"Absolute Vector: {self.unknown_vector}\n")
                self.unknown_vector_text.insert(tk.END, f"Оршовський S1: {s1_vector}\n")
                self.unknown_vector_text.insert(tk.END, f"Оршовський M1: {m1_vector}\n")

    def classify_image(self):
        if not self.unknown_vector:
            messagebox.showerror("Error", "No unknown image loaded.")
            return

        s1_vector = self.normalize_s1(self.unknown_vector)
        m1_vector = self.normalize_m1(self.unknown_vector)

        classification_results = ""
        for class_name in ['1', '2', '8']:
            s1min = self.class_stats[class_name]['S1MIN']
            s1max = self.class_stats[class_name]['S1MAX']
            m1min = self.class_stats[class_name]['M1MIN']
            m1max = self.class_stats[class_name]['M1MAX']

            s1_result = all(s1min[i] <= s1_vector[i] <= s1max[i] for i in range(len(s1_vector)))
            m1_result = all(m1min[i] <= m1_vector[i] <= m1max[i] for i in range(len(m1_vector)))

            classification_results += f"Class {class_name}:\n"
            classification_results += f"Оршовський S1 Match: {'Yes' if s1_result else 'No'}\n"
            classification_results += f"Оршовський M1 Match: {'Yes' if m1_result else 'No'}\n"

        self.unknown_vector_text.insert(tk.END, "\nClassification Results:\n")
        self.unknown_vector_text.insert(tk.END, classification_results)


if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureExtractionApp(root)
    root.mainloop()
