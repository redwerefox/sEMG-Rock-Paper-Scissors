import sys
import numpy as np
import tensorflow as tf
from ast import literal_eval as make_tuple

##############################################################################################################
###                                   TENSORFLOW Neural Net via Estimator                                  ###
###                        Input EMG data at time t as matrix (16,8), stores model hardcoded                ###
###                                    (16,8) matrix are never mixed with their gesture at time t:t+15       ###
##############################################################################################################

DIM = 16
		
class neuralNetwork :
	
	def __init__ (self, modelpath):
		
		###  specify hidden models and modelfunction : fully dense vs convoluted model
		self.COLUMNS = tf.feature_column.numeric_column("EMG",shape=(DIM,8),dtype=tf.int32)
		self.classifier = tf.estimator.Estimator(model_fn=sEMG_CNN_model_fn,params={"feature_columns":self.COLUMNS,
															"hidden_units":[64], "n_classes" : 3}
															,model_dir = modelpath)
															
	def train (self, batch_size, num_epochs):
		for k in range(num_epochs):
			for i in range(10):
				### Load datasets each and train it per epoch. 
				# load data
				path_train = "./datasets/123Dataset"+ str(i+1)+".csv"
				features, labels = read_dataset_file(path_train)
				input_size = len(labels)
				print(input_size)
				trainsteps = input_size//batch_size + 1
				input_fn = tf.estimator.inputs.numpy_input_fn(features,labels,num_epochs=1,shuffle=True)		
				
				
				
				# Train the Model.
				self.classifier.train(
				input_fn=input_fn,
				steps=trainsteps)
			
		self.evaluate(batch_size)
		
	def evaluate (self, batch_size):
		path_eval = "./datasets/123Dataseteval.csv"   ### Evaluation Dataset
		features, labels = read_dataset_file(path_eval)
		input_size = len(labels)
		print(input_size)
		evalsteps = input_size//batch_size + 1
		input_fn = tf.estimator.inputs.numpy_input_fn(features,labels,num_epochs=1,shuffle=True)
		
		self.classifier.evaluate(input_fn=input_fn,
		steps=evalsteps)

	def predict(self, input):   # input = one (8,8) EMG data
		input = predict8x8(input)
		input = { "EMG" : input}
		input_fn = tf.estimator.inputs.numpy_input_fn(input,num_epochs=1,shuffle=False)
		predictions = list(self.classifier.predict(input_fn=input_fn))
		return predictions
			
	
### Parse a single dataset line into required feature, label format
def read_dataset_file(csv_path):
	with open(csv_path,"r") as openfileobj:
		
		features = []
		labels = []
		
		#pack into lists
		for line in openfileobj:	
			feature, label = parse_line(line)
			features.append(feature)
			labels.append(label)
	
		features,labels = collect16x8 (features, labels)
		print(features.shape,labels.shape)
		
	#return xDict, y	
	return 	{"EMG" : (features)}, np.array(labels)
	
def parse_line (line):

	linetuple = make_tuple(line)
	#print(linetuple[0:2])
			
	feature = list(linetuple[0])
			
	label = linetuple[1] - 1
	
	#print(feature,label)
	
	return feature,label
####################################################
###  Should be only used in context with 2D data ###
####################################################	
def sEMG_CNN_model_fn(features,labels,mode,params):

	################ Network Archtitecture ######################################################################
	input_layer = tf.feature_column.input_layer(features,params['feature_columns'])
	input_layer = tf.reshape(input_layer, [-1,DIM,8,1])
	conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[3,3],padding="same",activation=tf.nn.relu)
	#pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2], strides=2)
	#conv2 = tf.layers.conv2d(inputs=pool1,filters=9,kernel_size=[3,3],padding="same",activation=tf.nn.relu)
	#pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2], strides=2)
	pool_flat = tf.reshape(conv1, [-1,DIM*8*64])
	dense = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)
	dp1 = tf.layers.dropout(dense,0.25)
	dense2 = tf.layers.dense(inputs=dense, units=512, activation=tf.nn.relu)
	dp2 = tf.layers.dropout(dense2,0.25)
	dense3 = tf.layers.dense(inputs=dense2, units=512, activation=tf.nn.relu)
	dp3 = tf.layers.dropout(dense3,0.25)
	#dense4 = tf.layers.dense(inputs=dense3, units=512, activation=tf.nn.relu)
	#dense5 = tf.layers.dense(inputs=dense4, units=512, activation=tf.nn.relu)
	logits = tf.layers.dense(inputs=dp3, units=params["n_classes"])
	##############################################################################################################
	
	# Compute predictions.
	predicted_classes = tf.argmax(logits, 1)
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'class_ids': predicted_classes[:, tf.newaxis],
			'probabilities': tf.nn.softmax(logits),
			'logits': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)
	
	####Eval & Train 
	# Compute loss
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),depth=params["n_classes"])
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
	
	# Compute evaluation metrics
	accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')

	
	metrics = {'accuracy': accuracy}
	tf.summary.scalar('accuracy', accuracy[1])

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=metrics)
		
	### Train
	#decay_steps = 100
	#lr_decayed = tf.train.cosine_decay(0.1,tf.train.get_global_step(),decay_steps)
	optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
	train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
	
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
	
####################################################
###  Fully dense . no filter                     ###
####################################################
def sEMG_model_fn(features,labels,mode,params):
	net = tf.feature_column.input_layer(features,params['feature_columns'])
		
	# Build the hidden layers, sized according to the 'hidden_units' param.
	for units in params['hidden_units']:
		net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
		#net = tf.layers.dropout(net, rate=0.0)
	
	# Compute logits (1 per class).
	logits = tf.layers.dense(net, params['n_classes'], activation=None)
	
	# Compute predictions.
	predicted_classes = tf.argmax(logits, 1)
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'class_ids': predicted_classes[:, tf.newaxis],
			'probabilities': tf.nn.softmax(logits),
			'logits': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)
	
	####Eval & Train 
	# Compute loss.
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
	
	# Compute evaluation metrics.
	accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')

	
	metrics = {'accuracy': accuracy}
	tf.summary.scalar('accuracy', accuracy[1])

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=metrics)
	
	### Train
	#decay_steps = 1000
	#lr_decayed = tf.train.cosine_decay(0.1,tf.train.get_global_step(),decay_steps)
	optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
	train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
	
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
			
#############################################
### Converting 16 tupels to 16x8 matrices   ###
### Rule : requires t=i=0:15 timestamps    ###
#############################################

def collect16x8 (features,labels):
	
	matrixFatures = []
	matrixLabels = []
	
	for idx,label in enumerate(labels):
		if all( i == label for i in labels[idx:idx+DIM]):
			feature = np.matrix(features[idx:idx+DIM])
			if feature.shape == (DIM,8):
				matrixFatures.append(feature)
				matrixLabels.append(label)
			
	matrixFatures = np.array(matrixFatures)
	print(matrixFatures.shape)
	
	return matrixFatures, np.array(matrixLabels)

### Prediction Override

def predict8x8 (features):
	
	matrixFatures = []
	
	feature = np.matrix(features[0:DIM])
	if feature.shape == (DIM,8):
		matrixFatures.append(feature)
			
	matrixFatures = np.array(matrixFatures)
	print(matrixFatures.shape)
	
	return matrixFatures
	
def main():			
	
	network = neuralNetwork ("./workspace20/model_C_F64P0_drop.25")
	#steps caluclated for train anyway
	network.train(batch_size=100,num_epochs=5)	

	#debugEMG = { "EMG" : [[-1, 0, 0, 1, 15, 6, 0, 5]]}
	#predictions = list(classifier.predict(input_fn=lambda:predict_input_fn(debugEMG,[5],1)))
	#print(predictions)

if __name__ == "__main__":
	main()