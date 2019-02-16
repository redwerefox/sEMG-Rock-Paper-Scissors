import os, requests
from ast import literal_eval
import kivy.resources 
from kivy.app import App
from kivy.properties import NumericProperty,ReferenceListProperty, ObjectProperty
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.graphics import *
from kivy.uix.anchorlayout import *
from kivy.uix.gridlayout import *
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock

from random import randint, shuffle

from MyoPoller import *
from sEMGRecorder import *
from gesture import *
from config import Config
from myo import init
from DataUtils import *

### global
TENSOR_CONNECTED = True
LEAP_CONNECTED = True
MYO_CONNECTED = True
select = InstructionGroup() # holds sets of canvas drawings
globals = {"myoInput":False,"playbutton":None,"responce":None,"lastTrain":0}

if MYO_CONNECTED:
	myo = MyoPoller()
	
# Screen for Gameplay. Myo Input or Button Input possible
# One Game of RoShamBo is started by pressing the play button	
class RoShamBoGame(Screen):
	background = ObjectProperty(None)
	selectionMenu = ObjectProperty(None)
	
# Screen for Training. Start training sessions and save data by stoping
# Also Update of model choosable
class RoShamBoTrain(Screen):
	background = ObjectProperty(None)
	trainSelection = ObjectProperty(None)
	
	
class SelectionMenu(AnchorLayout):
	
	########### Game ##########################################################
	# Pressed Button Callbacks
	#sets choice for player , remove existing choice Highlights and sets new one 
	def callbackRock(self):
		globals["myoInput"] = False
		self.choice = Gesture(0)
		self.canvas.remove(select)
		select.clear()
		select.add(Color(0,1,0,0.4))
		select.add(Rectangle(pos=self.ids["rock"].pos, size=self.ids["rock"].size))
		self.canvas.add(select)
		print("Rock pressed")
	def callbackPaper(self):
		globals["myoInput"] = False
		self.choice = Gesture(1)
		self.canvas.remove(select)
		select.clear()
		select.add(Color(0,1,0,0.4))
		select.add(Rectangle(pos=self.ids["paper"].pos, size=self.ids["paper"].size))
		self.canvas.add(select)
		print("Paper pressed")
	def callbackScissors(self):
		globals["myoInput"] = False
		self.choice = Gesture(2)
		self.canvas.remove(select)
		select.clear()
		select.add(Color(0,1,0,0.4))
		select.add(Rectangle(pos=self.ids["scissors"].pos, size=self.ids["scissors"].size))
		self.canvas.add(select)
		print("Scissors pressed")
	def callbackMyo(self):
		globals["myoInput"] = True
		self.canvas.remove(select)
		select.clear()
		select.add(Color(0,1,0,0.4))
		select.add(Rectangle(pos=self.ids["myo"].pos, size=self.ids["myo"].size))
		self.canvas.add(select)
	
	# GamePlay	Schedules RoShamBo countdown
	def callbackPlay(self):
		
		if not globals["myoInput"] :
			# reset choice
			self.canvas.remove(select)
			select.clear()
			self.choice = None 
	
		#remove Play-Start button from UI
		globals["playbutton"] = self.ids["play"]
		self.ids["center"].remove_widget(globals["playbutton"])
		
		#Schedule RoShamBo sequence
		Clock.schedule_once(self.callbackRo)
		Clock.schedule_once(self.callbackSham,1)
		Clock.schedule_once(self.callbackBo,2)
		Clock.schedule_once(self.callbackEval,2.33)
	
	# Schedule Callbacks
	def callbackRo(self,dt):
		self.gameText = Button(text="Ro",disable=True,size_hint_x=0.2,size_hint_y=0.2)
		self.ids["center"].add_widget(self.gameText)
		print("Ro")
	
	def callbackSham(self,dt):
		self.gameText.text = "Sham"
		print("Sham")
	
	def callbackBo(self,dt):
		self.gameText.text = "Bo"
		print("Bo")
	
	def callbackEval(self,dt):
		self.ids["center"].remove_widget(self.gameText)
		
		# evaluate Game
		enemy = Gesture(randint(0,2))
		if globals["myoInput"]:
			if MYO_CONNECTED and myo.Ready():  #8x8 is ready
				emg =  myo.emgDIMx8
				app = App.get_running_app()
				responce = requests.get("http://127.0.0.1:5000/myo/"+ str(app.config.map["user"])  + "/" + str(emg))
				responce = literal_eval(responce.text)
				self.choice = Gesture(responce[0])
		
		#display cpu
		self.cpu = Image(source=enemy.getCPUGestureImage())
		self.cpu.size_hint_x = 0.3
		self.cpu.size_hint_y = 0.3
		self.cpu.allow_stretch = True
		self.cpu.opacity = 0.9
		self.ids["up"].add_widget(self.cpu)
		
		#display game result
		self.img = None
		if self.choice:
			self.result = self.choice.beats(enemy)
			print(self.result)
			app = App.get_running_app()
			if self.result == "WIN":
				self.img = Image(source="WIN.png")
				app.config.map["Wins"] += 1
			if self.result == "DRAW":
				self.img = Image(source="draw.png")
				app.config.map["Draws"] += 1
			if self.result == "LOSS":
				self.img = Image(source="lose.png")
				app.config.map["Losses"] += 1
			self.img.size = self.ids["center"].size
			self.img.size_hint_y = 0.45
			self.img.size_hint_x = 0.45
			self.img.allow_stretch = True
			self.ids["center"].add_widget(self.img)
			
			
		else :
			print("Player wasn't ready")
		
		
		
		#Clock.schedule_once(self.callbackResult)
		Clock.schedule_once(self.callbackClearScreen,3)
	
	def callbackResult(self, dt):
		pass

	def callbackClearScreen(self,dt):
		
		# reset choice
		if not globals["myoInput"]:
			self.canvas.remove(select)
			select.clear()
			self.choice = None 
		
		if self.img:
			self.ids["center"].remove_widget(self.img)
		self.ids["center"].add_widget(globals["playbutton"])
		self.ids["up"].remove_widget(self.cpu)
		
	def callbackPause(self):
		sm.current = 'Pause'

########## Train ##########################################################
	# Freestyle and adaptive train
	# connects to Myo + leapmotion (requires python 2.7) installed myo-python
class TrainSelection (AnchorLayout):
	
	def callbackTrain (self):
		self.trainLoop = None
		if not MYO_CONNECTED:
			print("myo not ready")
			return
	
		app = App.get_running_app()
		self.num_dataset = app.config.map["countDatasets"] + 1
		filename = "record" + str(self.num_dataset) + ".csv"
		
		if LEAP_CONNECTED:
			self.recorder = app.recorder
			self.recorder.startRecording(app.dataset_dir+filename)
		
		self.startbutton = self.ids["start"]
		self.menubutton = self.ids["menu"]
		
		#alter buttons
		self.ids["box"].remove_widget(self.startbutton)
		self.ids["box"].remove_widget(self.menubutton)
		self.stopbtn = Button(text='Stop Training', size_hint_x=0.3, size_hint_y=0.25)
		self.stopbtn.bind(on_press=self.callbackTrain_stop)
		self.ids["box"].add_widget(self.stopbtn)
		
		# schedule game loop and initialization	
		Clock.schedule_once(self.beginTrain, 0)
		
	def callbackTrain_stop(self,value):
		if LEAP_CONNECTED:
			self.recorder.stopRecording()
			app = App.get_running_app()
			app.config.map["countDatasets"]  = self.num_dataset
		self.ids["box"].add_widget(self.startbutton)
		self.ids["box"].add_widget(self.menubutton)
		self.ids["box"].remove_widget(self.stopbtn)
		self.ids["up"].remove_widget(self.cpu)
		
		
	def beginTrain(self, dt):
		dt = 0
		gestures = [0,1,2]
		shuffle(gestures)
		nextGesture = gestures[0]
		if nextGesture == globals["lastTrain"]:
			nextGesture = gestures[1]
			
		self.enemy = Gesture(nextGesture)
		globals["lastTrain"] = nextGesture
		
		self.cpu = Image(source=self.enemy.getCPUGestureImage())
		self.cpu.size_hint_x = 0.3
		self.cpu.size_hint_y = 0.3
		self.cpu.allow_stretch = True
		self.cpu.opacity = 0.9
		self.ids["up"].clear_widgets()
		self.ids["up"].add_widget(self.cpu)
		
		if self.trainLoop:
			Clock.unschedule(self.trainUpdate, 2.5)
		self.trainLoop = Clock.schedule_interval(self.trainUpdate, 2.5)
			
	def trainUpdate (self, dt):		
		gesture_idx = self.recorder.gesture -1 #-1 to shift away defaults
		player = Gesture(gesture_idx)
		if player.beats(self.enemy) == "WIN" :
			print("nice !")
			Clock.unschedule(self.trainLoop)
			Clock.schedule_once(self.beginTrain)
		
	def callbackMain(self):
		app = App.get_running_app()
		data_dir = app.user_dir + "/datasets"
		dataUtils = DataUtils(data_dir)
		dataUtils.mergeFiles()	# creates datasets for NN training
		self.ids["up"].clear_widgets()
		sm.current = "Main"
	
	def callbackMainAndUpdateModel(self):
		app = App.get_running_app()
		data_dir = app.user_dir + "/datasets"
		dataUtils = DataUtils(data_dir)
		dataUtils.mergeFiles()	# creates datasets for NN training
		sm.current = "Info"
		
		
		
	
###################### Menu classes #################	
# Screens that hold simple Layouts
class PauseScreen(Screen):
	pass
		
class MainMenu(Screen):
	pass

class LoginGrid (Screen):
	pass
	
class ModelInfo(Screen):
	
	#wait for server responce in on thread to prevent UI freeze
		
	def callbackGoToMain(self, dt):
		sm.current = "Main"
	
	
	def runModelUpdate(self, dt):
	
		app = App.get_running_app()
		
		responce = requests.get("http://127.0.0.1:5000/user/"+ str(app.config.map["user"])  +  "/training" )
		globals['responce'] = float(responce.text.split(":")[1])
		print("Accuracy of model:" + str(globals['responce']))
		app.config.map["Accuracy"] = globals['responce']
		
	
	def startModelUpdate(self):
		
		if globals['responce']: # hack dont do training twice
			sm.current = "Main"
			button = self.ids["loading"]
			button.text = "Start Update"
			globals['responce'] = None
			return
		
		print(self.ids)
		print(self)
		button = self.ids["loading"]
		button.text = "Updating model..."
		button.disable = True
		
		print("Thread is here")
		Clock.schedule_once(self.runModelUpdate, 0)
		Clock.schedule_interval(self.waitForResponce, 1.0) 
			
		
	def waitForResponce(self, dt):
		if globals['responce']:
			button = self.ids["loading"]
			button.text = "Update done New Accuracy  " +  str(globals['responce'])
			button.disable = False
		
	
	##################### Background Classes ############
class RoShamBoBackground(Widget):
	pass
	
		
sm = ScreenManager()	
	
class RoShamBoApp(App):

	# inizialisation of App
	def build(self):
	
		if MYO_CONNECTED:
			myo.startListening()
		if LEAP_CONNECTED and MYO_CONNECTED:
			self.recorder = sEMGRecorder(myo)
	
		kivy.resources.resource_add_path("./assets")
		
		
		self.user = "default"
		self.user_dir = ""
		self.config = None
		
		### add screens to SCREEN MANAGER ###
		sm.add_widget(PauseScreen(name="Pause"))
		sm.add_widget(MainMenu(name="Main"))
		sm.add_widget(RoShamBoTrain(name="Train"))
		sm.add_widget(RoShamBoGame(name="Game"))
		sm.add_widget(LoginGrid(name="User"))
		sm.add_widget(ModelInfo(name="Info"))
		sm.current='User'		
		
		return sm
	
	# User profile generation
	def checkForUserProfile (self,user):
		dir = "./models/" + str(user)
		print(dir)
		if not os.path.exists(dir):
			os.makedirs(dir)
		return dir
	
	def checkForUserDatasets (self,user_dir):
		data_dir = user_dir + "./datasets/"
		print(data_dir)
		if not os.path.exists(data_dir):
			os.makedirs(data_dir)
		temp_dir = user_dir + "./training_temp/"
		if not os.path.exists(temp_dir):
			os.makedirs(temp_dir)
		temp_dir = user_dir + "./training/"
		if not os.path.exists(temp_dir):
			os.makedirs(temp_dir)
		return data_dir
	
	
	### User Login Menu ######
	
	def inputUser(self, value):
		self.user = value
	
	def loginUser(self):
		self.user_dir = self.checkForUserProfile(self.user)
		self.dataset_dir = self.checkForUserDatasets(self.user_dir)
		self.config = Config(self.user_dir)
		self.config.map["user"] = self.user
		self.userAcc = self.config.map["Accuracy"]
		#init tensorflow server               
		requests.get("http://127.0.0.1:5000/user/"+ str(self.config.map["user"])  + "/" + str(self.config.map["Accuracy"]))
		
		sm.current = "Main"
	#############################
		
	def on_stop(self):
		if MYO_CONNECTED:
			myo.stopListening()
		self.config.saveSession()
	
	
if __name__ == '__main__':
	RoShamBoApp().run()