import numpy as np
import cv2
import os, time, operator
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
from spellchecker import SpellChecker
from tensorflow.keras.models import model_from_json


class Application:
    def __init__(self):
        self.hs = SpellChecker()
        self.vs = cv2.VideoCapture(0)

        # Load models
        with open("Models/model_new.json", "r") as f:
            self.loaded_model = model_from_json(f.read())
        self.loaded_model.load_weights("Models/model_new.h5")

        with open("Models/model-bw_dru.json", "r") as f:
            self.loaded_model_dru = model_from_json(f.read())
        self.loaded_model_dru.load_weights("Models/model-bw_dru.h5")

        with open("Models/model-bw_tkdi.json", "r") as f:
            self.loaded_model_tkdi = model_from_json(f.read())
        self.loaded_model_tkdi.load_weights("Models/model-bw_tkdi.h5")

        with open("Models/model-bw_smn.json", "r") as f:
            self.loaded_model_smn = model_from_json(f.read())
        self.loaded_model_smn.load_weights("Models/model-bw_smn.h5")

        self.ct = {ch: 0 for ch in ascii_uppercase}
        self.ct['blank'] = 0
        self.blank_flag = 0

        self.current_symbol = "Empty"
        self.prev_symbol = None
        self.current_word = ""
        self.final_sentence = ""
        self.word_start_time = time.time()
        self.word_buffer = []

        self.setup_ui()
        self.video_loop()

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.geometry("900x900")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root, text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))
        self.T.place(x=60, y=5)

        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)

        self.T1 = tk.Label(self.root, text="Character :", font=("Courier", 30, "bold"))
        self.T1.place(x=10, y=540)

        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)

        self.T2 = tk.Label(self.root, text="Word :", font=("Courier", 30, "bold"))
        self.T2.place(x=10, y=595)

        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)

        self.T3 = tk.Label(self.root, text="Sentence :", font=("Courier", 30, "bold"))
        self.T3.place(x=10, y=645)

    def video_loop(self):
        ret, frame = self.vs.read()
        if ret:
            frame = cv2.flip(frame, 1)
            x1, y1, x2, y2 = int(0.5 * frame.shape[1]), 10, frame.shape[1] - 10, int(0.5 * frame.shape[1])
            roi = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(thresh)

            imgtk1 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)))
            imgtk2 = ImageTk.PhotoImage(image=Image.fromarray(thresh))

            self.panel.imgtk = imgtk1
            self.panel.config(image=imgtk1)

            self.panel2.imgtk = imgtk2
            self.panel2.config(image=imgtk2)

            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.current_word, font=("Courier", 30))
            self.panel5.config(text=self.final_sentence, font=("Courier", 30))

        self.root.after(10, self.video_loop)

    def predict(self, image):
        image = cv2.resize(image, (128, 128)).reshape(1, 128, 128, 1)

        result = self.loaded_model.predict(image)
        dru = self.loaded_model_dru.predict(image)
        tkdi = self.loaded_model_tkdi.predict(image)
        smn = self.loaded_model_smn.predict(image)

        pred = {'blank': result[0][0]}
        for i, ch in enumerate(ascii_uppercase):
            pred[ch] = result[0][i + 1]
        pred = sorted(pred.items(), key=operator.itemgetter(1), reverse=True)
        symbol = pred[0][0]

        if symbol in "DRU":
            pred_dru = {'D': dru[0][0], 'R': dru[0][1], 'U': dru[0][2]}
            symbol = sorted(pred_dru.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        elif symbol in "DIKT":
            pred_tkdi = {'D': tkdi[0][0], 'I': tkdi[0][1], 'K': tkdi[0][2], 'T': tkdi[0][3]}
            symbol = sorted(pred_tkdi.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        elif symbol in "MNS":
            pred_smn = {'M': smn[0][0], 'N': smn[0][1], 'S': smn[0][2]}
            symbol = sorted(pred_smn.items(), key=operator.itemgetter(1), reverse=True)[0][0]

        self.current_symbol = symbol

        # Time-based word building
        current_time = time.time()
        duration = current_time - self.word_start_time

        if symbol != 'blank':
            if self.prev_symbol != symbol:
                self.prev_symbol = symbol
                self.word_start_time = time.time()
            elif duration >= 1:
                self.word_buffer.append(symbol)
                print(f"Detected Letter: {symbol}")
                self.word_start_time = time.time()
                self.prev_symbol = None
        else:
            if duration >= 2 and self.word_buffer:
                word = ''.join(self.word_buffer)
                self.final_sentence += word + " "
                self.current_word = word
                print(f"Detected Word: {word}")
                self.word_buffer.clear()
                self.word_start_time = time.time()

    def destructor(self):
        print("Closing App")
        self.vs.release()
        cv2.destroyAllWindows()
        self.root.destroy()


print("Starting Application...")
Application().root.mainloop()
