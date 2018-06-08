############################################################
### Helpful relating Files / Datasets                    ###
############################################################
import sys
from ast import literal_eval as make_tuple


#####  GESTURES as index

### 0 = no gesture (deprecated)
### 1 = fist
### 2 = paper
### 3 = scissors


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