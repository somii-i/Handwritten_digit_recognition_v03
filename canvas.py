import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import os

class DigitRecognizerApp:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("Handwritten Digit Classifier")
        
        # Configure window
        self.window.geometry("400x500")
        self.window.resizable(False, False)
        #self.window.configure(bg='white')

        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Create main container frame
        main_frame = tk.Frame(self.window)
        main_frame.grid(row=0, column=0, sticky="nsew")
        #main_frame.configure(bg='white')
        
        # Configure main frame grid
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=0)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Create drawing canvas
        self.canvas = tk.Canvas(self.window, width=300, height=300, bg="white",relief='solid', borderwidth=2, cursor="cross")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        
        # Prediction label
        self.prediction_label = tk.Label(self.window, text="Draw a Digit", font=("Helvetica", 16, 'bold'))
        self.prediction_label.grid(row=1, column=0)
        
        # Buttons
        button_frame = ttk.Frame(self.window)
        button_frame.grid(row=2, column=0, pady=40)
        
        ttk.Button(button_frame, text="Predict", command=self.predict_digit).grid(row=0, column=0, padx=10)
        ttk.Button(button_frame, text="Clear", command=self.clear_canvas).grid(row=0, column=1, padx=10)
        
        # Initialize drawing context
        self.reset_drawing_context()
        self.brush_size = 15
        
        # Event bindings
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.reset_drawing)
        
        self.window.mainloop()
        
    def reset_drawing_context(self):
        """Properly reset both canvas and drawing context"""
        self.image = Image.new("L", (300, 300), "white")
        self.draw = ImageDraw.Draw(self.image)
        
    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-self.brush_size, y-self.brush_size,
                              x+self.brush_size, y+self.brush_size,
                              fill="black", outline="black", width=5)
        self.draw.ellipse([x-self.brush_size, y-self.brush_size,
                         x+self.brush_size, y+self.brush_size],
                        fill="black", width=5)
        
    def reset_drawing(self, event):
        self.draw = ImageDraw.Draw(self.image)
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.reset_drawing_context()
        self.prediction_label.config(text="Draw a Digit")
        
    def preprocess_image(self):
        img = self.image.resize((28, 28), Image.Resampling.BILINEAR)
        img = ImageOps.invert(img)
    
        img_array = np.array(img)
        non_zero = np.where(img_array > 0)
        if non_zero[0].size > 0 and non_zero[1].size > 0:
            x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
            y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
            centered = Image.new("L", (28, 28), 0)
            centered.paste(img.crop((x_min, y_min, x_max+1, y_max+1)),
                          (14 - (x_max - x_min)//2, 14 - (y_max - y_min)//2))
            img = centered
        
        img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0
        return img_array
        
    def predict_digit(self):
        try:
            processed_img = self.preprocess_image()
            prediction = self.model.predict(processed_img)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            self.prediction_label.config(
                text=f"Prediction: {digit} (Confidence: {confidence*100:.1f}%)"
            )
        except Exception as e:
            self.prediction_label.config(text=f"Error: {str(e)}")

if __name__ == "__main__":
    model_path = os.path.join("model", "saved_models", "digit_recognizer.h5")
    
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        app = DigitRecognizerApp(model)
    except Exception as e:
        print(f"Error loading model: {str(e)}")