from keras.models import load_model
from PIL import Image
import numpy as np

img_size = (128, 128)
batch_size = 15
model = load_model('model1_catsVSdogs_10epoch.h5')

results = {
    0: 'cat',
    1: 'dog'
}

im = Image.open("download3.jpg")
im = im.resize(img_size)
im = np.expand_dims(im, axis=0)
im = np.array(im)
im = im / 255
pred = model.predict([im])[0]
classes_x = np.argmax(pred, axis=0)
sign = results[classes_x]

print(
    "This image is %.2f percent %s"
    % (pred[0]*100, sign)
)
#print(pred)

# print(pred, sign)
