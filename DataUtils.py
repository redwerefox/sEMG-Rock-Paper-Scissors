############################################################
### Helpful relating Files / Datasets                    ###
############################################################
import sys, os
from ast import literal_eval as make_tuple


#####  GESTURES as index

### 0 = no gesture (deprecated)
### 1 = rock
### 2 = paper
### 3 = scissors

class DataUtils :
	
	def __init__ (self, modelpath):
		self.modelpath = modelpath
		print("operating on :",self.modelpath)

	# collect recordings, split them in 60,40% train, eval sets.
	# distribute equals gestures over all sets	
	def mergeFiles(self, percentage = 60):
		
		rocks = []
		papers = []
		scissors = []
		all = []	
		
		for file in os.listdir(self.modelpath):	
			
			file = self.modelpath + str("/") + file
			if file.endswith("csv") and "train" not in file and "eval" not in file:
				dataset = open(file,"r")
				for line in dataset:
					tuple = make_tuple(line)
					if tuple[1] == 1: #if gesture like rock 
						rocks.append(line)
					if tuple[1] == 2:
						papers.append(line)
					if tuple[1] == 3:
						scissors.append(line)
					all.append(line)
					
		#split dataset
		# percentage % = len * percentage/100
		train, eval = [], []
		
			
		train = all[:len(all)* percentage/100]
		eval = all[len(all)*percentage/100:]
		print("TrainSetsize ", len(train))
		print("EvalSetSize", len(eval))
				
		#train file
		train_dir = self.modelpath + str("/train.csv")
		train_file = open(train_dir,"w")
		for line in train:
			train_file.write(line)
		train_file.close()
		
		eval_dir = self.modelpath + str("/eval.csv")
		eval_file = open(eval_dir,"w")
		for line in eval:
			eval_file.write(line)
		eval_file.close()
	
def CountDistributions (filepath):
	
	file = open (filepath, "r")
	
	zeros = 0
	ones = 0
	twos = 0
	threes = 0
	
	
	for line in file:
		tuple = make_tuple(line)
		if tuple[1] == 0:
			zeros += 1
		elif tuple[1] == 1:
			ones += 1
		elif tuple[1] == 2:
			twos += 1
		elif tuple[1] == 3:
			threes += 1
	
	print ("File" + str(filepath) + "contains" + " :  " , zeros,ones,twos,threes)
	return zeros,ones,twos,threes

def main():
	CountDistributions(str(sys.argv[1]))

if __name__ == '__main__':
	main()