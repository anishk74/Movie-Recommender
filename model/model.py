import numpy as np
import tensorflow as tf
import time

class Recommender:
	def __init__(self,movies,users):
		self.n=10
		self.Theta=np.random.rand(users,self.n)
		self.X=np.random.rand(movies,self.n)

	def fit(self,Y,R,epochs=20,alpha=0.00001,validation_split=0.2):
		
		history=[]
		
		total_ratings=sum(sum(R))
		for i in range(epochs):
			#start=time.time()

			cost=(1/(2*total_ratings)) * sum(sum((np.matmul(self.X, self.Theta.T) * R - Y)**2)) # ((X*Theta')*R - Y)^2
			#end=time.time()
			#print('Cost',end-start)
			history.append(cost)
			X_grad=np.zeros(self.X.shape)
			#start=time.time()
			for i in range(len(self.X)):
				X_grad[i] = np.matmul(
									(np.matmul(
										self.X[i],
										self.Theta[R[i,:]==1].T) - Y[i,R[i,:]==1]),
									self.Theta[R[i,:]==1])

			Theta_grad=np.zeros(self.Theta.shape)
			for j in range(len(self.Theta)):
				Theta_grad[j] = np.matmul(
									(np.matmul(
										self.X[R[:,j]==1],
										self.Theta[j].T) - Y[R[:,j]==1,j]),
									self.X[R[:,j]==1])
			#end=time.time()
			#print('Gradient',end-start)
			self.X = self.X - alpha * X_grad
			self.Theta = self.Theta - alpha * Theta_grad

		

		return history



