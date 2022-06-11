import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np

from keras.models import load_model

model = load_model('model1_catsVSdogs_10epoch.h5')
file_path = ''
classes = {
    0: 'Image is of Cat',
    1: 'Image is of Dog',
}
# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Cats_vs_Dogs_Classification')
top.configure(background='black')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def prediction_image():
    global label_packed, file_path

    open_img = Image.open(file_path)
    reshape_image = open_img.resize((128, 128))
    image = np.expand_dims(reshape_image, axis=0)
    img_arr = np.array(image)
    img_arr = img_arr / 255
    prediction = model.predict([img_arr])[0]
    classes_x = np.argmax(prediction, axis=0)
    sign = classes[classes_x]
    print(sign)
    label.configure(foreground='#011638', text=sign)


def show_prediction_button(file_path):
    pred_b = Button(top, text="Predict Image", command=lambda: prediction_image(file_path), padx=50, pady=15)
    pred_b.configure(background='white', foreground='blue', font=('arial', 10, 'bold'))
    pred_b.place(relx=0.79, rely=0.46)


def to_image_upload():
    try:
        global file_path
        file_path = filedialog.askopenfilename()
        upload = Image.open(file_path)
        upload.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        image = ImageTk.PhotoImage(upload)
        sign_image.configure(image=image)
        sign_image.image = image
        label.configure(text='')
        # show_prediction_button(file_path)
    except:
        pass


upload_b1 = Button(top, text="Image Upload", command=to_image_upload, padx=10, pady=5)
upload_b1.configure(background='white', foreground='blue', font=('arial', 10, 'bold'))
upload_b1.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)


upload_b2 = Button(top, text="Predict Image", command = prediction_image, padx=10, pady=5)
upload_b2.configure(background='white', foreground='blue', font=('arial', 10, 'bold'))
upload_b2.pack(side=RIGHT, pady=50)
upload_b2.place(relx=0.79, rely=0.46)



heading = Label(top, text="Cats and Dogs Predictions", pady=50, font=('arial', 20, 'bold'))
heading.configure(background='black', foreground='blue')
heading.pack()
top.mainloop()
