from __future__ import print_function

import sys
import time

sys.path.append('../Myo-raw')

from common import *
from myo_raw import MyoRaw
from myo_raw import Pose

import threading
from threading import Thread

class MyoListener (MyoRaw):
	HIST_LEN=25

	def __init__(self, tty=None):
		MyoRaw.__init__(self,tty)
		self.add_emg_handler(self.emg_handler)
		self.emg = None
		
		
	def emg_handler(self, emg, moving):
		self.emg = emg
		#print('emg:', emg, '| moving:', moving)
	
	def startListening(self, filepath):
		file = open(filepath,"w")
		
	def startListening(self):
	
		self.thread_stop = threading.Event()
		self.thread = Thread(target=self.listen)
		self.thread.start()
		

	def stopListening(self):
		self.thread_stop.set()
		self.thread.join()
		


	def listen(self):
		self.connect()
		self.vibrate(2)
		
    #self.connect()
		while not self.thread_stop.is_set():
			self.run()
			
			self.listen()
	#def listen(self):
		#self.connect()
	#	self.run()
	#	return self.emg
	
	#def stopListening(self):
	#	self.disconnect()
	
	
	def runListeningThread(self):
		try:
			while True:
				self.run()
				self.emg = myoListener.emg
				file.writelines(str(self.emg))
		
		except:
			KeyboardInterrupt
		finally:
			self.disconnect()
			file.close()
			

def main():

	#debug
	filename = "./emgdata.csv"
	file = open(filename,"w")

	
	myoListener = MyoListener()
	myoListener.connect()
	myoListener.vibrate(2)
	
	print("start loop")
	try :
		while True:
			myoListener.run()
			emg = myoListener.emg
			file.writelines(str(emg))
			
			
	except KeyboardInterrupt:
		pass
	finally:
		myoListener.disconnect()
		file.close()
		
if __name__ == '__main__':
	main()