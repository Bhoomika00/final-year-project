import sklearn.externals
import joblib
from pip._vendor import certifi
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from PIL import Image
#import pymongo[srv]
#import dnspython


from flask import Flask, render_template, request, url_for, session, redirect, flash
from flask_pymongo import PyMongo
import bcrypt
#import dns
from pymongo import MongoClient
app = Flask(__name__)
client = MongoClient("mongodb+srv://root_user:root123@cluster0.88pft.mongodb.net/test"
,tlsCAFile=certifi.where())
db = client.get_database('parkinsonsusers')
users = db.users



@app.route('/',methods=['GET'])
def landing():
    return render_template("register.html")
@app.route('/dashboard/spiral', methods=['GET'])
def home1():
    btnname = "spiral"
    return render_template("test.html", btnname=btnname)



@app.route('/doctors',methods=['GET'])
def doc():
    return render_template('doctors.html')
@app.route('/dashboard/wave', methods=['GET'])
def home2():
    btnname = "wave"
    return render_template("test.html", btnname=btnname)
@app.route('/home', methods=['GET'])
def home():
    #btnname = "wave"
    return render_template("dashboard.html")






@app.route('/dashboard', methods=['GET'])
def dashboard():
    imgname = session['username']


    if('filenameSpiral' in session.keys()):
        imgname = imgname.replace(' ', '-').lower()
        finalname1 = "static/images/spiralTest/" + imgname + ".png"
    else:
        finalname1 = "/"

    if('filenameWave' in session.keys()):
        imgname = imgname.replace(' ', '-').lower()
        finalname2 = "static/images/waveTest/" + imgname + ".png"
    else:
        finalname2 = "/"
    imglist = [finalname1, finalname2]
    wavetest = None
    spiraltest = None

    if('a' in session.keys()):
        wavetest = session['a']

    if('s' in session.keys()):
        spiraltest = session['s']

    if(wavetest is None and spiraltest is None):
        report = "Not tested yet"
    if(wavetest == "healthy" and spiraltest == "healthy"):
        report = "Dear "+session['username']+" you are completely healthy"
    elif(wavetest == "parkinson" and spiraltest == "parkinson"):
        report = "Dear "+session['username']+" you are detected positive for parkisons we advice you to concern with doctor as soon as possible."

    elif (wavetest == "parkinson" and spiraltest == "healthy"):
        report = "Dear "+session['username']+" we detected that you may have chances to develope parksisons in future if problem continues please visit hospital."

    elif (wavetest == "healthy" and spiraltest == "parkinson"):
        report = "Dear "+session['username']+" we detected that you may have chances to develope paralysis in future if problem continues please visit hospital."

    else:
        report = "Not tested yet"

    imglist.append(report)
    return render_template("dash.html", imglist=imglist)

@app.route('/login',methods=['POST'])
def login():
    loginUser = users.find_one({"name": request.form['username']})

    if loginUser:
        if bcrypt.hashpw(request.form['pass'].encode('utf-8'), loginUser["password"]) == loginUser["password"]:
            session['username'] = request.form['username']

            return redirect(url_for('home'))
        return render_template("register.html")
    return render_template("register.html")



@app.route('/logout')
def logout():
    session.pop('username',None)
    session.pop('a',None)
    session.pop('s',None)
    session.pop('filenameSpiral',None)
    session.pop('filenameWave',None)
    return redirect(url_for('landing'))

@app.route('/logoutdash')
def logoutdash():
    session.pop('username',None)
    return redirect(url_for('landing'))


@app.route('/register',methods=['POST', "GET"])
def register():
    if request.method == 'POST':

        existing_user = users.find_one({'name':request.form['username']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['pass'].encode('utf-8'), bcrypt.gensalt())
            users.insert_one({'name': request.form['username'], 'password': hashpass})
            session['username'] = request.form['username']
            return redirect(url_for('home'))
        else:
            flash("user already exists")

    return render_template('register.html')



## Machine Learning Algorithm
def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features


def load_split(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        # [healthy, healthy, parkinson, ....]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))

        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        features = quantify_image(image)
        data.append(features)
        labels.append(label)


    return (np.array(data), np.array(labels))








def classify_my_image_spiral(image_path,mdl):
    image = cv2.imread(image_path)
    output = image.copy()
    output = cv2.resize(output, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)
    model = joblib.load(mdl)
    preds = model.predict([features])


    #label = le.inverse_transform(preds)[0]
    label = "parkinson" if preds[0] else "healthy"
    session['s'] = label
    color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # show the output montage
    imgname = session['username']
    imgname = imgname.replace(' ', '-').lower()
    cv2.imwrite("static/images/spiralTest/"+imgname+".png", output)
    session['filenameSpiral'] = "static/images/spiralTest/"+imgname + ".png"
    return label


@app.route("/dashboard/spiral", methods=["POST"])
def predictSpiral():

    spiralmodel=os.path.join("models/", "random_forest_spiral_model.pkl")
    imagefile = request.files['imagefile']
    imagepath = "./static/images/spiralTest/"+imagefile.filename
    imagefile.save(imagepath)

    session["spiralTestResult"] = classify_my_image_spiral(imagepath,spiralmodel)
    #output=classify_my_image_spiral(imagepath,spiralmodel)
    btnname = "spiral"


    return redirect(url_for('home'))

########wave prediction
def classify_my_image_wave(image_path,mdl):
    image = cv2.imread(image_path)
    output = image.copy()
    output = cv2.resize(output, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)
    model2 = joblib.load(mdl)
    preds = model2.predict([features])


    #label = le1.inverse_transform(preds)[0]
    label = "parkinson" if preds[0] else "healthy"
    session['a'] = str(label)
    color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # show the output montage
    imgname = session['username']
    imgname = imgname.replace(' ', '-').lower()
    cv2.imwrite("static/images/waveTest/"+imgname+".png", output)
    session['filenameWave'] = "static/images/waveTest/"+imgname + ".png"


@app.route("/dashboard/wave", methods=["POST"])
def predictWave():
    imagefile = request.files['imagefile']
    wavemodel=os.path.join("models/", "random_forest_wave_model.pkl")
    imagepath = "./static/images/waveTest/"+imagefile.filename
    imagefile.save(imagepath)
    classify_my_image_wave(imagepath,wavemodel)

    return redirect(url_for('home'))









if __name__ == "__main__":
    app.secret_key = 'mysecret'
    app.run(debug=True)

