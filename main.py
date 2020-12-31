import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

from model.model import Recommender

sns.set()

ds=loadmat('dataset.mat')

Y=ds['Y']	#Actual ratings
R=ds['R']	#Mask matrix for ratings

no_movies,no_users=Y.shape

model=Recommender(no_movies,no_users)

history=model.fit(Y,R,epochs=100,alpha=0.0001)

plt.plot(history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('img/output.png')
plt.show()

