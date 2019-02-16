###########
# Imports #
###########
import os, sys, inspect
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = '../lib/x64' if sys.maxsize > 2**32 else '../lib/x86'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))
import Leap
import thread, time

##################
#  FINGER_TYPES  #
##################
TYPE_THUMB = 0
TYPE_INDEX = 1
TYPE_MIDDLE = 2
TYPE_RING = 3
TYPE_PINKY = 4

#####  GESTURES as index

### 0 = no gestureAsLabel
### 1 = fist
### 2 = paper
### 3 = scissors


### Listen to Leap Motion device, updates current frame with callback ### 
class LeapListener (Leap.Listener):

	######  Callbacks  ##################################################
	
	def onInit(self,controller):
		print("Initialzed")
	
	### informs about functioning device
	def on_connect(self, controller):
		print("Leap Motion Connected")
	
	def on_frame(self, controller):
		self.frame = controller.frame()
	
	#######################################################################
	
	def startListening(self):
		self.controller = Leap.Controller()
		self.controller.add_listener(self)
	
	def stopListening(self):
		self.controller.remove_listener(self)
		
	
	def getGestures(self):
		self.frame = self.controller.frame()
		#print("frame id:", frame.id)
		self.getExtendedFingers(self.firstHand())
		
		fist = self.detectFist()
		palm = self.detectPalm()
		scissors = self.detectScissors()
				
		#print("detect fist ? ",fist)
		#print("detect paper ? ",palm)
		#print("detect scissors ? ",scissors)
	
		gestures = fist,palm,scissors
		
		return self.gestureAsLabel(gestures)
		
	### returns first hand tracked in scene 
	def firstHand(self):
		return self.frame.hands[0]
	
	def getFinger(self, hand, index):
		pointables = hand.pointables
		for pointable in pointables:
			if pointable.is_finger:
			#check for fingertype
				finger = Leap.Finger(pointable)
				if finger.type == index:
					return finger
		return None
	
	### returns tuple first element shows if measure is valid, secound list of extended fingers
	def getExtendedFingers (self, hand):
		if hand.is_valid :
			thumb = self.getFinger(hand,TYPE_THUMB)
			index = self.getFinger(hand, TYPE_INDEX)
			middle = self.getFinger(hand, TYPE_MIDDLE)
			ring = self.getFinger(hand, TYPE_RING)
			pinky = self.getFinger(hand, TYPE_PINKY)
			if thumb.is_valid and index.is_valid and middle.is_valid and ring.is_valid and pinky.is_valid:
				thumbExtend = thumb.is_extended
				indexExtend = index.is_extended
				middleExtend = middle.is_extended
				ringExtend = ring.is_extended
				pinkyExtend = pinky.is_extended
				#print("Pinches:",thumbExtend,indexExtend,middleExtend,ringExtend,pinkyExtend)
				return (True, (thumbExtend,indexExtend,middleExtend,ringExtend,pinkyExtend))
		return (False, (False,False,False,False,False))  
	
	### Fist
	def detectFist(self):
		hand = self.firstHand()
		handInfo = self.getExtendedFingers(hand)
		if handInfo[0] :
			fingerinfo = handInfo[1]
			# all fingers are closed
			if  not( fingerinfo[0] or fingerinfo[1] or fingerinfo[2] or fingerinfo[3] or fingerinfo[4]) and hand.grab_strength == 1 :
				return True
		return False
	
	def detectPalm(self):
		hand = self.firstHand()
		handInfo = self.getExtendedFingers(hand)
		if handInfo[0] :
			fingerinfo = handInfo[1]
			# all fingers are closed
			if (fingerinfo[0] and fingerinfo[1] and fingerinfo[2] and fingerinfo[3] and fingerinfo[4]) :
				return True
		return False
	
	def detectScissors(self):
		hand = self.firstHand()
		handInfo = self.getExtendedFingers(hand)
		if handInfo[0] :
			fingerinfo = handInfo[1]
			# all fingers are closed 
			if (fingerinfo[1] and fingerinfo[2] and not fingerinfo[3] and not fingerinfo[4]) :
				return True
		return False
	
	### For NN's we need integer labels
	### 0 = no gestureAsLabel
	### 1 = fist
	### 2 = paper
	### 3 = scissors
	def gestureAsLabel(self, gestures):
		if gestures[0]:
			return 1
		if gestures[1]:
			return 2
		if gestures[2]:
			return 3
		return 0
		
	def debugGrab_Strength (self):
		hand = self.firstHand()
		return hand.grab_strength
	 
	def handConfidence (self):
		hand = self.firstHand()
		return hand.confidence
	
def main():
	print("Kitsune on the run")
	leapListener = LeapListener()
	leapListener.startListening()	
		
	try:
		while True :
			sys.stdin.readline()
			gestures = leapListener.getGestures()
	except KeyboardInterrupt:
		pass
	finally:
		leapListener.stopListening()

### Program Entry Point ###
def debug ():

	print("Fox on the run")
	controller = Leap.Controller()
	listener = LeapListener()
	
	controller.add_listener(listener)
	
	#Keep process running till input
	try:
		while True :
			sys.stdin.readline()
			frame = listener.frame
			print("frame id:", frame.id)
			listener.getExtendedFingers(listener.firstHand())
			
			#print("grab_strength:", listener.debugGrab_Strength())
			#print("confidence: ",listener.handConfidence())
			
			print("detect fist ? ",listener.detectFist())
			print("detect paper ? ",listener.detectPalm())
			print("detect scissors ? ",listener.detectScissors())
	except KeyboardInterrupt:
		pass
	#finally :
		#controller.remove_listener(listener)
	 
		
if __name__ == "__main__":
	debug()