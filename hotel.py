import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor as KNN    
california = datasets.fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, random_state=100)

california_arr=np.concatenate((X_train,y_train.reshape(-1,1)), axis=1)
california_pd=pd.DataFrame(california_arr, columns=[*california.feature_names, 'MEDV'])
fig,ax = plt.subplots(figsize=(9,5))
sns.heatmap(california_pd.corr().round(2), annot=True, square=True, ax=ax)
#plt.show()

scaler=StandardScaler()
scaler.fit(X_train)
X_train_fs_scaled=scaler.transform(X_train)
X_test_fs_scaled=scaler.transform(X_test)

knn = KNN(n_neighbors=50, metric='manhattan')
knn.fit(X_train_fs_scaled, y_train)
y_test_predictions = knn.predict(X_test_fs_scaled)
print('The test R^2 is: {:.2f}'.format(knn.score(X_test_fs_scaled, y_test)))


