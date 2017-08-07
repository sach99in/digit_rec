import numpy as np

from  flask import Flask, render_template, request
import cPickle as pickle 
from scipy.misc import imsave, imread, imresize

import re

#system level operations (like loading files)
import sys 
#for reading operating system data
import os
from PIL import Image
import numpy as np
import cv2



my_model = pickle.load(open("digit.pkl","rb"))
app = Flask(__name__)

global model, graph
#initialize these variables
#model, graph = init()
img = cv2.imread("output.png", 0)
img1 =img.clone().reshape(1,1)


@app.route('/')
def index():
 	return render_template('index.html')




@app.route('/predict/',methods=['GET','POST'])
def predict():
	
	
	#perform the prediction
	out = my_model.predict(img1)
	print(out)
	print(np.argmax(out,axis=1))
	
	#convert the response to a string
	response = np.array_str(np.argmax(out,axis=1))
	return response	
	

if __name__ == "__main__":
	app.run()