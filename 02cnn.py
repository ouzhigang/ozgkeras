import os, sys
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical

if __name__ == "__main__":
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	
	x_train /= 255
	x_test /= 255
	
	y_train = to_categorical(y_train, 10)
	y_test = to_categorical(y_test, 10)
	
	model = None
	if os.path.exists("./02cnn.h5"):
		model = load_model("./02cnn.h5")
	else:
		model = Sequential()
		model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
		model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
		model.add(MaxPooling2D(pool_size = (2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation = 'relu'))
		model.add(Dropout(0.5))
		model.add(Dense(10, activation = 'softmax'))
		
		model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
		model.fit(x_train, y_train, epochs = 8, batch_size = 128)
	
		model.save("./02cnn.h5")
	
	#测试误差率、准确率
	#res = model.evaluate(x_test, y_test)
	#print(res)
	
	#输出训练结果
	res = model.predict_classes(x_test)
	print(res[4])
	print(y_test[4])
	