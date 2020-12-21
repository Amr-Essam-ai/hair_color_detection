
import os
import cgi  # A CGI script is invoked by an HTTP server, usually to process user input submitted through an HTML <FORM>
import cv2
import json
import falcon
import numpy as np 
from numpy import fliplr
from falcon_cors import CORS 
import efficientnet.tfkeras as efn 
from tensorflow.keras.models import load_model
from utils import predict_ , predict_TTA, load_model_




api = application = falcon.API()

def load_trained_model():
    global model
    model = load_model_()
    return model



def convert_image(images,w=224,h=224):
    images_j=[]
    c=3
    for i in range (len(images)):
        img = cv2.imdecode(np.fromstring(images["image"+str(i)].file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        img =cv2.resize(img, (w,h))
        img = np.expand_dims(img, axis =0)
        if len(img.shape) <3:c=1
        data = (img).reshape(1, w, h, c)
        images_j.append(data)

    return np.vstack(images_j)

def convert_image_TTA(images,w=224,h=224):
    images_j=[];fliplrim=[]
    c=3
    for i in range (len(images)):
        img = cv2.imdecode(np.fromstring(images["image"+str(i)].file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        img =cv2.resize(img, (w,h))
        img = np.expand_dims(img, axis =0)
        if len(img.shape) <3:c=1
        data = (img).reshape(1, w, h, c)
        images_j.append(data)
        fliplrim.append(fliplr(data))

    images_j.extend(fliplrim)

    return np.vstack(images_j)




class PredictResource(object):

    def __init__(self, model):
        self.model = model

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.body = 'hair_color_api '

    def on_post(self, req, resp):
        data={}

        form = cgi.FieldStorage(fp=req.stream, environ=req.env)
       
        images=convert_image(form,w=224,h=224)

        data["result"] = predict_(images, model)

        resp.body = json.dumps(data, ensure_ascii=False)

predict = PredictResource(model=load_trained_model())
api.add_route('/hair_color_api/', predict)
