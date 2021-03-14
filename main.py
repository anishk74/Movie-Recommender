import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

import tensorflow as tf
from model.model import Recommender

sns.set()

ds=loadmat('dataset.mat')

Y=ds['Y'].astype(float)	#Actual ratings
R=ds['R'].astype(float)	#Mask matrix for ratings

plt.title('No. of movies rated by a User')
plt.hist(np.sum(R, axis=0), bins=25) #no. of movies rated by a user
plt.axvline(np.sum(R, axis=0).mean(), linestyle='dashed', color='g')
plt.savefig('img/users_hist.png')
plt.close()

plt.title('No. of ratings a movie get')
plt.hist(np.sum(R, axis=1), bins=25) #no. of ratings a movie got
plt.axvline(np.sum(R, axis=1).mean(), linestyle='dashed', color='g')
plt.savefig('img/movies_hist.png')
plt.close()

Y[R!=1] = 0

meanY = np.sum(Y,axis=1,keepdims=True)/np.sum(R,axis=1,keepdims=True)
Y = Y-meanY
Y[R!=1] = 0

no_movies,no_users=Y.shape

model=Recommender(no_movies,no_users)

history=model.fit(Y,R,epochs=500,alpha=0.05)

plt.plot(history[0],linewidth=0.75)
plt.plot(history[1],linewidth=0.75)

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.ylim(0.2,1.6)
plt.legend(['Regularised Loss','Actual Loss'])
plt.savefig('img/lossoutput.png')
plt.close()

