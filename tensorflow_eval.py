import os, inspect, sys, time

import threading
from threading import Thread

src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = '../lib/x64' if sys.maxsize > 2**32 else '../lib/x86'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

from MyoPoller import *
from neuralNetwork import *
from neuralNetworkAdvanced import *
#####################################################################
###         Loads Model and feeds it with EMG data to predict     ###
###                                                               ###
#####################################################################


#####  GESTURES as index

### 0 = fist
### 1 = paper
### 2 = scissors
class EMG_Predicter:
	
	def __init__ (self,modelpath,userpath):
	
		self.myo = MyoPoller()
		self.myo.startListening()

		self.neuralNet = neuralNetwork(modelpath,userpath)
	
	def predict(self):
		if self.myo.Ready() :  #8x8 is ready
			emg8x8 = self.myo.emg8x8
			prediction = self.neuralNet.predict(emg8x8)
			for predict in prediction:
				if "class_ids" in predict.keys():
					return predict["class_ids"]
	
	def stop(self):
		self.myo.stopListening()

		
def main():
	emg_predicter = EMG_Predicter("./models/Kitsune/training","./models/Fox/dataset")
	
	#Keep process running till input
	try:
		filename = "./comparison2_G4.txt"
		file = open(filename,"w")
	
		while True :
			sys.stdin.readline()
			if emg_predicter.myo.Ready() :  #8x8 is ready
				emg8x8 = emg_predicter.myo.emgDIMx8
				prediction = emg_predicter.neuralNet.predict(emg8x8)
				file.write(str(prediction) + "\n")
				print(prediction)
			
	except KeyboardInterrupt:
		pass
	finally :
		emg_predicter.stop()
	 
		
if __name__ == "__main__":
	main()