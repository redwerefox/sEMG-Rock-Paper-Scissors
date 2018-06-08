import sys
import numpy as np
import tensorflow as tf
import pywt
from ast import literal_eval as make_tuple

##############################################################################################################
###                                   TENSORFLOW Neural Net via Estimator                                  ###
###                        Input EMG data at time t as matrix (16,8), stores model hardcoded                ###
###                                    (16,8) matrix are never mixed with their gesture at time t:t+15       ###
##############################################################################################################

DIM = 24
DIM_MODEL = 29
stats = {}
		
class neuralNetworkAdvanced :
	
	def __init__ (self, modelpath,userpath):
		
		self.learning_rate = 0.1 #default
		
		###  specify hidden models and modelfunction : fully dense vs convoluted model
		self.COLUMNS = tf.feature_column.numeric_column("EMG",shape=(DIM_MODEL,8),dtype=tf.int32)
		self.classifier = tf.estimator.Estimator(model_fn=sEMG_CNN_model_fn,params={"feature_columns":self.COLUMNS,
															"hidden_units":[64,128,512], "n_classes" : 3, "nn":self}
															,model_dir = modelpath)
		self.userpath = userpath
		
		
	def train (self, batch_size, num_epochs):
		for k in range(num_epochs):
			for i in range(1):
				### Load datasets each and train it per epoch. 
				# load data
				
				path_train = self.userpath +  "/train"+".csv"
				features, labels = read_dataset_file(path_train)
				input_size = len(labels)
				print(input_size)
				trainsteps = input_size//batch_size + 1
				input_fn = tf.estimator.inputs.numpy_input_fn(features,labels,num_epochs=1,shuffle=True)		
				
				
				
				# Train the Model.
				self.classifier.train(
				input_fn=input_fn,
				steps=trainsteps)
			
		return self.evaluate(batch_size)
		
		
	# evaluates the trained network with eval set and returns dict containing acc + loss	
	def evaluate (self, batch_size):
		path_eval =  self.userpath + "./eval.csv"   ### Evaluation Dataset
		features, labels = read_dataset_file(path_eval)
		input_size = len(labels)
		print(input_size)
		evalsteps = input_size//batch_size + 1
		input_fn = tf.estimator.inputs.numpy_input_fn(features,labels,num_epochs=1,shuffle=True)
			
		specs = (self.classifier.evaluate(input_fn=input_fn,
		steps=evalsteps))
		
		# printing information about training process
		print("Accuarcy evaluation & loss {}".format(specs))
		
		return specs

	def predict(self, input):   # input = one (8,8) EMG data
		input = predictWavelet(input)
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
	
		features,labels = collectWavelets (features, labels)
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
	input_layer = tf.reshape(input_layer, [-1,DIM_MODEL,8,1])
	conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[3,3],padding="same",activation=tf.nn.relu)
	#print(conv1.shape)
	#pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2], strides=2)
	#conv2 = tf.layers.conv2d(inputs=pool1,filters=9,kernel_size=[3,3],padding="same",activation=tf.nn.relu)
	#pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2], strides=2)
	pool_flat = tf.reshape(conv1, [-1,DIM_MODEL*8*64])
	dp0 = tf.layers.dropout(pool_flat,0.0)
	dense = tf.layers.dense(inputs=dp0, units=512, activation=tf.nn.relu)
	dp1 = tf.layers.dropout(dense,0.25,training=mode == tf.estimator.ModeKeys.TRAIN)
	dense2 = tf.layers.dense(inputs=dp1, units=512, activation=tf.nn.relu)
	dp2 = tf.layers.dropout(dense2,0.25,training=mode == tf.estimator.ModeKeys.TRAIN)
	dense3 = tf.layers.dense(inputs=dp2, units=128, activation=tf.nn.relu)
	dp3 = tf.layers.dropout(dense3,0.25,training=mode == tf.estimator.ModeKeys.TRAIN)
	"""
	#pool_flat_nor =tf.layers.batch_normalization(inputs=pool_flat)
	dense = tf.layers.dense(inputs=pool_flat, units=50, activation=tf.sigmoid)
	dp1 = tf.layers.dropout(inputs=dense,rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
	pool_flat_nor =tf.layers.batch_normalization(inputs=dp1)
	dense2 = tf.layers.dense(inputs=pool_flat_nor, units=512, activation=tf.nn.sigmoid)
	dp2 = tf.layers.dropout(inputs=dense2,rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
	pool_flat_nor_2 =tf.layers.batch_normalization(inputs=dp2)
	dense3 = tf.layers.dense(inputs=pool_flat_nor_2, units=128, activation=tf.nn.sigmoid)
	dp3 = tf.layers.dropout(inputs=dense3,rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
	"""
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
	
	#add regulariation
	vars = tf.trainable_variables()
	
	"""
	lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 0
	loss += lossL2
	"""
	# Compute evaluation metrics
	accuracy= tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')
							   				
	metrics = {'accuracy': accuracy}
	scalar = tf.summary.scalar('accuracy', accuracy[1])
	
	
	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=metrics)
		
	### Train
	#decay_steps = 100
	#lr_decayed = tf.train.cosine_decay(0.1,tf.train.get_global_step(),decay_steps)
	optimizer = tf.train.AdagradOptimizer(learning_rate=params['nn'].learning_rate)
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
	optimizer = tf.train.AdagradOptimizer(learning_rate=0.05)
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


def collectFrequencys (features,labels):
	
	matrixFatures = []
	matrixLabels = []
	
	for idx,label in enumerate(labels):
		if all( i == label for i in labels[idx:idx+DIM]):
			feature = np.matrix(features[idx:idx+DIM])
			if feature.shape == (DIM,8):
			
				freq_feature = feature[:,:,np.newaxis]
				frequencys = np.empty([8,16], dtype=complex)
				#fast fourier to add fequency domain
				for i in range(8):
				
					#ith column
					time_signal = feature[:,i]
					frequency = np.fft.fft(time_signal)
					frequency = np.reshape(frequency, [16])
					frequencys[i] = frequency
				
				frequencys=np.reshape(frequencys,[16,8])
				matrixFatures.append(frequencys)
				matrixLabels.append(label)

				
	matrixFatures = np.array(matrixFatures)
	print(matrixFatures.shape)
	
	
	return matrixFatures, np.array(matrixLabels)	

def collectWavelets (features,labels):
	
	matrixFatures = []
	matrixLabels = []
	
	for idx,label in enumerate(labels):
		if all( i == label for i in labels[idx:idx+DIM]):
			feature = np.matrix(features[idx:idx+DIM])
			if feature.shape == (DIM,8):
			
				freq_feature = feature[:,:,np.newaxis]
				channels = np.empty([8,DIM_MODEL])
				#fast fourier to add fequency domain
				for i in range(8):
				
					#ith column
					values = []
					time_signal = np.reshape(feature[:,i], [DIM])
					cA2, cD2, cD1 = pywt.wavedec(time_signal, 'db2', level=2)
					cA2 = cA2.flatten()
					cD2 = cD2.flatten()
					cD1 = cD1.flatten()
					[values.append(v) for v in cA2]
					[values.append(v) for v in cD2]
					[values.append(v) for v in cD1]
					
					
					#wavelet = np.append(cA2, cD2, cD1)
					channels[i] = values
					
				matrixFatures.append(channels)
				matrixLabels.append(label)

				
	matrixFatures = np.array(matrixFatures)

	
	
	return matrixFatures, np.array(matrixLabels)	


	
### Prediction Override

def predictWavelet (features):
	
	matrixFatures = []
	
	feature = np.matrix(features[0:DIM])
	if feature.shape == (DIM,8):
		freq_feature = feature[:,:,np.newaxis]
		channels = np.empty([8,DIM_MODEL])
		#fast fourier to add fequency domain
		for i in range(8):
				
			#ith column
			values = []
			time_signal = np.reshape(feature[:,i], [DIM])
			cA2, cD2, cD1 = pywt.wavedec(time_signal, 'db2', level=2)
			cA2 = cA2.flatten()
			cD2 = cD2.flatten()
			cD1 = cD1.flatten()
			[values.append(v) for v in cA2]
			[values.append(v) for v in cD2]
			[values.append(v) for v in cD1]
					
					
			#wavelet = np.append(cA2, cD2, cD1)
			channels[i] = values
					
		matrixFatures.append(channels)

			
	matrixFatures = np.array(matrixFatures)
	print(matrixFatures.shape)
	
	return matrixFatures
	
def main():			
	
	network = neuralNetworkAdvanced ("./wavespace/modelLV2_re_ADA_dr.dym_delay_3lay", "./datasets")
	#steps caluclated for train anyway
	network.learning_rate = 0.025
	for i in range (50):
		specs = network.train(batch_size=500,num_epochs=1)	
		accuracy = specs["accuracy"]
		
	#debugEMG = { "EMG" : [[-1, 0, 0, 1, 15, 6, 0, 5]]}
	#predictions = list(classifier.predict(input_fn=lambda:predict_input_fn(debugEMG,[5],1)))
	#print(predictions)

if __name__ == "__main__":
	main()