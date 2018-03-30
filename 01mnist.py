import os, sys
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical

if __name__ == "__main__":
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	#转换为二维
	x_train = x_train.reshape(60000, 784)
	x_test = x_test.reshape(10000, 784)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	#把值转换为0到1之间
	x_train /= 255
	x_test /= 255

	#归类，共10个数
	num_classes = 10
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test, num_classes)
	
	model = None
	if os.path.exists("./mnist.h5"):
		model = load_model("./mnist.h5")
	else:
		model = Sequential()
		model.add(Dense(10, input_shape = (784,), activation = 'softmax'))

		model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
		model.fit(x_train, y_train, epochs = 8, batch_size = 128)

		model.save('./mnist.h5')

	#测试误差率、准确率
	#res = model.evaluate(x_test, y_test)
	#print(res)

	#输出训练结果
	res = model.predict_classes(x_test, batch_size = 128)
	print(res[3])
	print(y_test[3])