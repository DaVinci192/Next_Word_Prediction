# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 07:35:55 2021

@author: Randhir
TEST flask app - template code taken from https://morioh.com/p/bbbc75c00f96
"""

from flask import Flask
app = Flask(__name__)

from flask import render_template
from flask import request

from wtforms import (Form, TextField, validators, SubmitField, 
DecimalField, IntegerField)

class ReusableForm(Form):
    """User entry form for entering specifics for generation"""
    # Starting seed
    sentence = TextField("Enter the beginngings of a sentence!:", validators=[
                     validators.InputRequired()])
    # Submit button
    submit = SubmitField("Enter")


# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    # Create form
    form = ReusableForm(request.form)

    # Send template information to index.html
    return render_template('index.html', form=form)

# Create form
    form = ReusableForm(request.form)

    # On form entry and all conditions met
    if request.method == 'POST' and form.validate():
        sentence = request.form['sentence']

app.run(host='0.0.0.0', port=50000)

