import os, inspect, sys, time

src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = '../lib/x64' if sys.maxsize > 2**32 else '../lib/x86'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

from LeapListener import *
from MyoPoller import *

######################################################
###  SEMG Recorder : Record loop for TrainingData  ###
###  Currently 0's not tracked                     ###
######################################################

#####  GESTURES as index

### 0 = no gesture (deprecated)
### 1 = fist
### 2 = paper
### 3 = scissors


class sEMGRecorder():
	
	### Connect devices
	def __init__ (self, myo):
		myo = myo
		leap = LeapListener()
		
		self.myoListener = myo
		self.leapListener = leap
		
		self.gesture = 0
		
		
	### Start Listeners
	def startRecording(self, filepath):	
		filename = filepath      
		self.file = open(filename,"w")
	
	
		self.leapListener.startListening()
		#self.myoListener.startListening()
		self.startThread()

		
	def stopRecording(self):
		self.leapListener.stopListening()
		#self.myoListener.stopListening()
		self.stopThread()
		self.file.close()
	
	### Record loop 
	### First UserInput starts recording , second ends it.
	### saves dataset to hardcoded filepath
	def record(self,filepath):
		try:
			print("Press Enter to start Recording")
			sys.stdin.readline()
			self.startRecording(filepath)
			print("Press Enter to quit")
			sys.stdin.readline()
			self.stopRecording()
			print("Recording stopped")
		except KeyboardInterrupt:
			self.stopRecording()
		finally:
			self.stopRecording()
		
	def startThread(self):
		self.thread_stop = threading.Event()
		self.thread = Thread(target=self.runRecording)
		self.thread.start()


	def stopThread(self):
		self.thread_stop.set()
		self.thread.join()
	
	# Reccord THREAD
	def runRecording(self):
		time.sleep(0.5)
		while not self.thread_stop.is_set() :
			if self.leapListener.controller.is_connected :
				self.gesture = self.leapListener.getGestures()
				if self.gesture != 0:  # remove 0's from record
					emg = self.myoListener.emg
					
					trainData = (emg,self.gesture,time.clock())
					self.file.write(str(trainData) + "\n")
					
	
def main ():
	
	recorder = sEMGRecorder()
	recorder.record("./datasets/210FreestyleRasmus3.csv")
		
if __name__ == '__main__':
	main()