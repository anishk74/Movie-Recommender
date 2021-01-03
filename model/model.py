import numpy as np
import tensorflow as tf
import time

class Recommender:
	def __init__(self,movies,users):
		self.n=10
		self.Theta=tf.Variable(np.random.rand(users,self.n))
		self.X=tf.Variable(np.random.rand(movies,self.n))

	def fit(self,Y,R,epochs=20,alpha=0.001):
		
		history=[]

		indices=np.asarray(np.where(R==1)).T
		ratings=Y[R==1]
		sparse_Y=tf.SparseTensor(indices=indices,
									values=ratings,
									dense_shape=Y.shape)

		loss = lambda: tf.losses.mean_squared_error(sparse_Y.values,
				tf.reduce_sum(
				tf.gather(self.X, sparse_Y.indices[:, 0]) *
				tf.gather(self.Theta, sparse_Y.indices[:, 1]),
				axis=1))

		opt=tf.keras.optimizers.Adam(learning_rate=alpha)
		
		for epoch in range(epochs):

			with tf.GradientTape() as tape:
				loss_value = loss()

			history.append(float(loss_value))

			grads = tape.gradient(loss_value, [self.X,self.Theta])
			opt.apply_gradients(zip(grads, [self.X,self.Theta]))
			if epoch%10==0:
				print('Epochs:', epoch, 'Loss:', float(loss_value),'\n')
		print('Epochs:', epoch, 'Loss:', float(loss_value),'\n')


		return history



