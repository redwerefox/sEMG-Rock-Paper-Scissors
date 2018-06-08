
import ast, os
##################################################################
### Config stores user specific information between sessions #####
##################################################################

class Config:
	
	# User profile
	def checkForUserConfig (self,dir):
		dir += str("/config.kcf")
		print(dir)
		if not os.path.isfile(dir):
			open(dir,"w")
			#defaults init
			self.first = True
		return dir
	
	
	def __init__ (self, filepath):
		self.first = False
		### read File
		self.map = {}
		configFile = open(self.checkForUserConfig(filepath),"r")
		self.dir = self.checkForUserConfig(filepath)
		if self.first: #INITIALIZATION
			self.map.update({"user":"filepath"})
			self.map.update({"countDatasets":0})
			self.map.update({"Wins":0})
			self.map.update({"Draws":0})
			self.map.update({"Losses":0})
			self.map.update({"Accuracy":0.0})
		else : 
			for line in configFile:
				self.map = ast.literal_eval(line)

		print(self.map)	
		configFile.close()

	def saveSession(self):
		configFile = open(self.dir,"w")
		print(self.map)
		configFile.write(str(self.map))
		configFile.close()