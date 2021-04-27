#It saves in .py format
import streamlit as st
import numpy as np
from skimage.io import imread #iuse pip install scikit-image if skimage does not work
from skimage.transform import resize
import pickle
from PIL import image
import matplotlib.pyplot as plt
st.title('Image Classifier Using Machine Learning')
st.text('Upload the Image')

model = pickle.load(open('img_model.p','rb'))
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img, caption='Uploaded Image')

  if st.button('Predict'):
    CATEGORIES = ['Single', 'Double', 'Triple', 'Zero']
    st.write('Result...')
    flat_data=[]
    img = np.array(img)
    img_resized = resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    #plt.imshow(img_resized)
    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    st.write(f'PREDICTED OUTPUT: {y_out}')
    q = model.predict_pro(flat_data)
    for index,item in enumerate(CATEGORIES):
      st.write(f'{item} : {q[0][index]*100}')
