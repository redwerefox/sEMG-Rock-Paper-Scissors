import sys
import threading
from threading import Thread


from myo import init, Hub, Feed, StreamEmg
################################################################################
###                     MyoPoller Connects device , polls data               ###
###                     Requires MyoConnect to run !                         ###
################################################################################

DIM = 24

class MyoPoller ():
	def __init__(self):
		self.ready = False
		self.emg = None
		
		init()
	
	def startListening(self):
		self.thread_stop = threading.Event()
		self.thread = Thread(target=self.listen)
		self.thread.start()
		
	def Ready (self):
		return self.ready
	
	
	def stopListening(self):
		print("Stop myo!!!!")
		self.thread_stop.set()
		self.thread.join()
	
	### Main Polling thread ###
	def listen (self):
		feed = Feed()
		hub = Hub()
		hub.run(1000, feed)
		try:
			myo = feed.wait_for_single_device(timeout=2.0)
			if not myo:
				print("No Myo connected after 2 seconds")
			print("Hello, Myo!")
			myo.set_stream_emg(StreamEmg.enabled)
			self.emgDIMx8 = []
			while hub.running and myo.connected and not self.thread_stop.is_set():
				quat = myo.orientation
				if self.emg != myo.emg:
					self.emg = myo.emg
					self.emgDIMx8.append(self.emg)
				###8 elements in feed , keep the last 8 and be ready iff
				if len(self.emgDIMx8) == DIM + 1:
					self.emgDIMx8.pop(0)
					self.ready = True
					
				#print('Orientation:', quat.x, quat.y, quat.z, quat.w)
				#print('EMG:', self.emg)
		finally:
			hub.stop(True)
			hub.shutdown()  # !! crucial	
			#stopListening()
	
def main():
	myopoller = MyoPoller()
	myopoller.startListening()
	
if __name__ == '__main__':
	main()