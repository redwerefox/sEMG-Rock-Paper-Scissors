class Gesture():
	
	###
	# default = -1
	# rock = 0
	# paper = 1
	# scissors = 2
	
	def __init__ (self, gest):
		self.gesture = gest

	def isRock(self):
		return self.gesture == 0
	
	def isPaper(self):
		return self.gesture == 1
		
	def isScissors(self):
		return self.gesture == 2	
	
	def beats(self,gest):
		if self.gesture == (gest.gesture + 1)%3:
			return "WIN"
		elif self.gesture == gest.gesture:
			return "DRAW"
		else :
			return "LOSS"
	def getCPUGestureImage (self):
		if self.gesture == 0:
			return "fist.png"
		if self.gesture == 1:
			return "palm.png"
		if self.gesture == 2:
			return "scissor.png"
		
def main ():

	# TEST CASES
	
	print("Case1 Rock beats paper")
	gestureA = Gesture(0)
	gestureB = Gesture(1)
	print("A is Rock:" , gestureA.isRock())
	print("B is Paper:" , gestureB.isPaper())
	print("A beats B ", gestureA.beats(gestureB))	
	
	print("Case2 Rock beats Scissors")
	gestureA = Gesture(0)
	gestureB = Gesture(2)
	print("A is Rock:" , gestureA.isRock())
	print("B is Paper:" , gestureB.isScissors())
	print("A beats B ", gestureA.beats(gestureB))
	
	print("Case1 Paper beats paper")
	gestureA = Gesture(1)
	gestureB = Gesture(1)
	print("A is Rock:" , gestureA.isPaper())
	print("B is Paper:" , gestureB.isPaper())
	print("A beats B ", gestureA.beats(gestureB))
	
	print("Case1 paper beats scissors")
	gestureA = Gesture(1)
	gestureB = Gesture(2)
	print("A is Rock:" , gestureA.isPaper())
	print("B is Paper:" , gestureB.isScissors())
	print("A beats B ", gestureA.beats(gestureB))
	
	print("Case1 scissors beats paper")
	gestureA = Gesture(2)
	gestureB = Gesture(1)
	print("A is Rock:" , gestureA.isScissors())
	print("B is Paper:" , gestureB.isPaper())
	print("A beats B ", gestureA.beats(gestureB))
	
	print("Case1 Scissors beats Rock")
	gestureA = Gesture(2)
	gestureB = Gesture(0)
	print("A is Rock:" , gestureA.isScissors())
	print("B is Paper:" , gestureB.isRock())
	print("A beats B ", gestureA.beats(gestureB))
	
	print("Case1 Rock beats Rock")
	gestureA = Gesture(0)
	gestureB = Gesture(0)
	print("A is Rock:" , gestureA.isRock())
	print("B is Paper:" , gestureB.isRock())
	print("A beats B ", gestureA.beats(gestureB))
	
if __name__ == '__main__':
	main()