import os, sys
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

if __name__ == "__main__":

	img = image.load_img('./1.jpg', target_size = (224, 224))
	model = VGG16(include_top = True, weights = 'imagenet')
	
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis = 0)
	x = preprocess_input(x)

	scores = model.predict(x)
	
	'''
	class_table = open('./synset_words.txt', 'r')
	lines = class_table.readlines()
	#print("scores type: ", type(scores))
	#print("scores shape: ", scores.shape)
	#print(np.argmax(scores))
	print('result is ', lines[np.argmax(scores)])
	class_table.close()
	'''
	
	print('Predicted:', decode_predictions(scores, top = 5)[0])
	