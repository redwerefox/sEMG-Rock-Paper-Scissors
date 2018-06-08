import os
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

from sEMGRecorder import *
#from tensorflow_eval_8x8 import *
from gesture import *
from config import Config
from myo import init

### global
TENSOR_CONNECTED = False
LEAP_CONNECTED = True
MYO_CONNECTED = True
select = InstructionGroup() # holds sets of canvas drawings


# Conncects to Pedictier Model
if TENSOR_CONNECTED:
	emg_predicter = EMG_Predicter("../tensorflow/workspace/model_17d_candidate")	
	
# holds Background and Layout as selectionMenu including Buttons	
class RoShamBoGame(Screen):
	background = ObjectProperty(None)
	selectionMenu = ObjectProperty(None)

class RoShamBoTrain(Screen):
	background = ObjectProperty(None)
	trainSelection = ObjectProperty(None)
	
class SelectionMenu(AnchorLayout):
	
	########### Game ##########################################################
	# Pressed Button Callbacks
	#sets choice for player , remove existing choice Highlights and sets new one 
	def callbackRock(self):
		self.myoInput = False
		self.choice = Gesture(0)
		self.canvas.remove(select)
		select.clear()
		select.add(Color(0,1,0,0.4))
		select.add(Rectangle(pos=self.ids["rock"].pos, size=self.ids["rock"].size))
		self.canvas.add(select)
		print("Rock pressed")
	def callbackPaper(self):
		self.myoInput = False
		self.choice = Gesture(1)
		self.canvas.remove(select)
		select.clear()
		select.add(Color(0,1,0,0.4))
		select.add(Rectangle(pos=self.ids["paper"].pos, size=self.ids["paper"].size))
		self.canvas.add(select)
		print("Paper pressed")
	def callbackScissors(self):
		self.myoInput = False
		self.choice = Gesture(2)
		self.canvas.remove(select)
		select.clear()
		select.add(Color(0,1,0,0.4))
		select.add(Rectangle(pos=self.ids["scissors"].pos, size=self.ids["scissors"].size))
		self.canvas.add(select)
		print("Scissors pressed")
	def callbackMyo(self):
		self.myoInput = True
		self.canvas.remove(select)
		select.clear()
		select.add(Color(0,1,0,0.4))
		select.add(Rectangle(pos=self.ids["myo"].pos, size=self.ids["myo"].size))
		self.canvas.add(select)
	
	# GamePlay	Schedules RoShamBo countdown
	def callbackPlay(self):
		
		if not MYO_CONNECTED :
			# reset choice
			self.canvas.remove(select)
			select.clear()
			self.choice = None 
	
		#remove Play-Start button from UI
		self.playButton = self.ids["play"]
		
		self.ids["center"].remove_widget(self.playButton)
		
		#Schedule RoShamBo sequence
		Clock.schedule_once(self.callbackRo)
		Clock.schedule_once(self.callbackSham,1)
		Clock.schedule_once(self.callbackBo,2)
		Clock.schedule_once(self.callbackEval,2.5)
	
	# Schedule Callbacks
	def callbackRo(self,dt):
		print("Ro")
	
	def callbackSham(self,dt):
		print("Sham")
	
	def callbackBo(self,dt):
		print("Bo")
	
	def callbackEval(self,dt):
	
		
	
		if not MYO_CONNECTED:
			self.myoInput = False
	
		# evaluate Game
		enemy = Gesture(randint(0,2))
		if self.myoInput:
			if TENSOR_CONNECTED:
				# Ugly !!! TOdo better emg_predict parsing !!
				emg_class = emg_predicter.predict()
				print(emg_class)
				self.choice = Gesture(emg_class[0])
		
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
			self.img.size_hint_y = 0.5
			self.img.size_hint_x = 0.5
			self.img.allow_stretch = True
			self.ids["center"].add_widget(self.img)
			
			
		else :
			print("Player wasn't ready")
		
		
		
		#Clock.schedule_once(self.callbackResult)
		Clock.schedule_once(self.callbackClearScreen,2)
	
	def callbackResult(self, dt):
		pass

	def callbackClearScreen(self,dt):
		
		# reset choice
		if not self.myoInput:
			self.canvas.remove(select)
			select.clear()
			self.choice = None 
		
		if self.img:
			self.ids["center"].remove_widget(self.img)
		self.ids["center"].add_widget(self.playButton)
		self.ids["up"].remove_widget(self.cpu)
		
	def callbackPause(self):
		sm.current = 'Pause'

########## Train ##########################################################
	# Freestyle and adaptive train
	# connects to Myo + leapmotion (requires python 2.7) installed myo-python
class TrainSelection (AnchorLayout):
	
	def callbackTrain (self):
		if not MYO_CONNECTED:
			print("myo not ready")
			return
	
		app = App.get_running_app()
		self.num_dataset = app.config.map["countDatasets"] + 1
		filename = "record" + str(self.num_dataset) + ".csv"
		
		if LEAP_CONNECTED:
			self.recorder = app.recorder
			self.recorder.startRecording(app.dataset_dir+filename)
		
		#alter buttons
		self.ids["box"].remove_widget(self.ids["start"])
		self.ids["box"].remove_widget(self.ids["menu"])
		self.stopbtn = Button(text='Stop Training', size_hint_x=0.3, size_hint_y=0.3)
		self.stopbtn.bind(on_press=self.callbackTrain_stop)
		self.ids["box"].add_widget(self.stopbtn)
		
		# schedule game loop and initialization	
		Clock.schedule_once(self.beginTrain, 0)
		
	def callbackTrain_stop(self,value):
		if LEAP_CONNECTED:
			self.recorder.stopRecording()
			app = App.get_running_app()
			app.config.map["countDatasets"]  = self.num_dataset
		self.ids["box"].add_widget(self.ids["start"])
		self.ids["box"].add_widget(self.ids["menu"])
		self.ids["box"].remove_widget(self.stopbtn)
		self.ids["up"].remove_widget(self.cpu)
		
		
	def beginTrain(self, dt):
		gestures = [0,1,2]
		shuffle(gestures)
		self.enemy = Gesture(gestures[0])
		
		self.cpu = Image(source=self.enemy.getCPUGestureImage())
		self.cpu.size_hint_x = 0.3
		self.cpu.size_hint_y = 0.3
		self.cpu.allow_stretch = True
		self.cpu.opacity = 0.9
		self.ids["up"].add_widget(self.cpu)
				

		self.trainLoop = Clock.schedule_interval(self.trainUpdate, 1.0/4.0)
			
	def trainUpdate (self, dt):		
		gesture_idx = self.recorder.gesture -1 #-1 to shift away defaults
		player = Gesture(gesture_idx)
		if player.beats(self.enemy) == "WIN" :
			print("nice !")
			self.ids["up"].remove_widget(self.cpu)
			Clock.unschedule(self.trainLoop)
			Clock.schedule_once(self.beginTrain)
		
	def callbackMain(self):
		sm.current = "Main"
 ###################### Menu classes #################	
class PauseScreen(Screen):
	pass
		
class MainMenu(Screen):
	pass

class LoginGrid (Screen):
	pass
	
	
	
	##################### Background Classes ############
class RoShamBoBackground(Widget):
	pass
	
	
	
sm = ScreenManager()	
	
class RoShamBoApp(App):

	def build(self):
		#init() # myo library
		if LEAP_CONNECTED:
			self.recorder = sEMGRecorder()
	
		kivy.resources.resource_add_path("./assets")
		### SCREEN MANAGER ###
		
		self.user = "default"
		self.user_dir = ""
		self.config = None
		
		sm.add_widget(PauseScreen(name="Pause"))
		sm.add_widget(MainMenu(name="Main"))
		sm.add_widget(RoShamBoTrain(name="Train"))
		sm.add_widget(RoShamBoGame(name="Game"))
		sm.add_widget(LoginGrid(name="User"))
		sm.current='User'		
		
		return sm
	
	# User profile
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
		return data_dir
	
	
	### User Login Menu ######
	def inputUser(self, value):
		self.user = value
	
	def loginUser(self):
		self.user_dir = self.checkForUserProfile(self.user)
		self.dataset_dir = self.checkForUserDatasets(self.user_dir)
		self.config = Config(self.user_dir)
		self.config.map["user"] = self.user
		sm.current = "Main"
	#############################
		
	def on_stop(self):
		if TENSOR_CONNECTED:
			emg_predicter.stop()
		self.config.saveSession()
	
	
if __name__ == '__main__':
	RoShamBoApp().run()