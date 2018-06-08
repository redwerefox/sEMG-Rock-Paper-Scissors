"""
Networking with flask
to run it do
> set FLASK_APP=TensorFlask.py
> flask run
"""	
from flask import Flask

class TensorFlask(Flask):

	def __init__(self,name):
		super(TensorFlask,self).__init__(name)

app  = TensorFlask(__name__)

@app.route('/')
def hello_flask():
	return("Kitsune in the flask")