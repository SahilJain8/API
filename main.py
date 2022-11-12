from flask import Flask,render_template,request,jsonify
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf

model=keras.models.load_model("Model/model.h5")
classes_name={0:"Choroidal neovascularization",1:"Diabetic macular edema ",2:"Drusen",3:"Normal"}


app=Flask(__name__)

@app.route('/')
def home():
    return "hello"



@app.route('/predection',methods = ['GET','POST'])
def predection():
    if request.method=="POST":
        file=request.files["image"]
        file.save(file.filename)
        img=tf.keras.utils.load_img(file.filename)
        os.remove(file.filename)
        img=tf.keras.utils.img_to_array(img)
        image=tf.image.resize(img,(160,160))
        image = np.expand_dims(image, axis=0)
        image=image/255.
        predection=model.predict(image)
        pre=predection.flatten()
        m=pre.max()
        pre=list(pre)
        return str(classes_name[pre.index(m)])   

     

        
if __name__=="__main__":
    app.run()
