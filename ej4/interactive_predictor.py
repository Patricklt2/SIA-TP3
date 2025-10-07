import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import os
import sys

# Add the project root to the Python path to find the 'perceptrons' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import create_mnist_mlp

class DigitDrawer(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("MNIST Interactive Predictor")
        self.model = model
        self.canvas_width = 280
        self.canvas_height = 280
        self.pen_width = 20

        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="black", cursor="cross")
        self.canvas.pack(pady=10, padx=10)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=10)

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.prediction_label = tk.Label(self, text="Draw a digit (0-9)", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)

        # Create a PIL image and a draw object to draw on
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - self.pen_width), (event.y - self.pen_width)
        x2, y2 = (event.x + self.pen_width), (event.y + self.pen_width)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a digit (0-9)")

    def predict_digit(self):
        # Resize the image to 28x28 and antialias it
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Invert and convert to numpy array
        img_array = np.array(img_resized)
        
        # Normalize the image data to be between 0 and 1
        img_normalized = img_array.astype('float32') / 255.0
        
        # Reshape for the model: (1, 784, 1)
        img_final = img_normalized.reshape(1, 28*28, 1)

        # Make a prediction
        prediction = self.model.predict(img_final)
        predicted_digit = np.argmax(prediction)

        self.prediction_label.config(text=f"Predicted Digit: {predicted_digit}")

if __name__ == "__main__":
    # Create the MLP model structure
    mnist_mlp = create_mnist_mlp()

    # Load the trained weights
    model_path = os.path.join(os.path.dirname(__file__), 'mnist_model.npz')
    try:
        mnist_mlp.load_weights(model_path)
    except FileNotFoundError:
        messagebox.showerror("Error", f"Model weights not found at '{model_path}'.\nPlease run 'python3 ej4/main.py' first to train and save the model.")
        sys.exit()

    # Create and run the drawing app
    app = DigitDrawer(model=mnist_mlp)
    app.mainloop()