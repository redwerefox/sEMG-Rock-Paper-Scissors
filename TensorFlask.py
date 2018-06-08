"""
Networking with flask
to run it do
> set FLASK_APP=TensorFlask.py
> flask run
"""	
from ast import literal_eval
from flask import Flask
from neuralNetwork import *
from neuralNetworkAdvanced import *

import shutil
		
app  = Flask(__name__)
neuralNet = None #default

@app.route('/')
def hello_flask():
	app.neuralNet = None
	return("Kitsune in the flask")

@app.route('/user/<user>/<acc>')	
def init_user(user, acc):	
		app.model_path = "./models/" + str (user) + "/training"
		app.temp_path = app.model_path + "_temp"
		app.user_path = "./models/" + str (user) + "/datasets"
		app.neuralNet = neuralNetwork(app.model_path,app.user_path)
		app.currentAccuracy = float(acc)
		return (str(user) + " has been initialized" )
		
		
@app.route('/user/<user>/training')	
def train_user(user):
	#learn on temp
	shutil.rmtree(app.temp_path)
	app.neuralNet = neuralNetwork(app.temp_path,app.user_path)
	app.neuralNet.learning_rate = 0.1
	for epochs in range (50):

			
		stats, train = app.neuralNet.train(batch_size=2000,num_epochs=1)
		accuracy = stats["accuracy"]
		if accuracy >= app.currentAccuracy: # update our model
			shutil.rmtree(app.model_path)
			shutil.copytree(app.temp_path,app.model_path)
			app.currentAccuracy = accuracy
		
	#set back on best model_path
	app.neuralNet = neuralNetwork(app.model_path,app.user_path)
	return(str("Done with accuracy :") + str (app.currentAccuracy))

@app.route('/user/<user>/trainingadvanced')	
def train_user_advanced(user):
	#learn on temp
	#shutil.rmtree(app.temp_path)
	app.neuralNet = neuralNetworkAdvanced(app.temp_path,app.user_path)
	app.neuralNet.learning_rate = 0.025
	for epochs in range (10):

		stats = app.neuralNet.train(batch_size=2000,num_epochs=1)
		accuracy = stats["accuracy"]
		if accuracy >= app.currentAccuracy : # update our model
			shutil.rmtree(app.model_path)
			shutil.copytree(app.temp_path,app.model_path)
			app.currentAccuracy = accuracy
		
	#set back on best model_path
	app.neuralNet = neuralNetworkAdvanced(app.model_path,app.user_path)
	return(str("Done with accuracy :") + str (app.currentAccuracy))

@app.route('/myoAdvanced/<user>/<emg>')
def predict_advanced(user, emg):

	emg_data = literal_eval(emg)
	prediction = app.neuralNet.predict(emg_data)
	for predict in prediction:
		if "class_ids" in predict.keys():
			return str(predict["class_ids"])
	return "0"
	
@app.route('/myo/<user>/<emg>')
def predict(user, emg):

	emg_data = literal_eval(emg)
	prediction = app.neuralNet.predict(emg_data)
	for predict in prediction:
		if "class_ids" in predict.keys():
			return str(predict["class_ids"])
	return "0"
	