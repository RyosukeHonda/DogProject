from flask import Flask,Response
from wtforms import SelectField,SubmitField,Form
from flask import render_template,request
import cv2
import sys,os
from dog_classifier import *

test_path = "static/test_images"
app = Flask(__name__)

class SelectForm(Form):
    select = SelectField('',choices=[(test_path+"/cat.jpg","Cat1"),
                                    (test_path+"/dog1.jpg","Dog1"),
                                    (test_path+"/dog2.jpg","Dog2"),
                                    (test_path+"/dog3.jpg","Dog3"),
                                    (test_path+"/dog4.jpg","Dog4"),
                                    (test_path+"/human.jpg","Human1"),
                                    (test_path+"/human2.jpg","Human2"),
                                    (test_path+"/other.jpg","Other")])

@app.route('/')
def index():
    form = SelectForm(request.form)
    return render_template('dog_test.html',form = form)

@app.route('/results',methods=['POST'])
def results():
    form = SelectForm(request.form)
    if request.method =="POST":

        img_path = request.form["select"]
        res = detector(img_path)

        return render_template('result.html',results = res,path = img_path)
    return render_template('dog_test.html')

if __name__ == '__main__':
    app.run(debug=True)