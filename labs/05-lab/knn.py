class KNN:

    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.distance_matrix = None
    
    def train(self):
        self.distance_matrix = []
        for i in range(len(self.X_train)):
            l = []
            for j in range(len(self.X_train)):
                if i==j:
                    l.append(0)
                else:   
                    l.append(np.linalg.norm(self.X_train[i] - self.X_train[j]))
            self.distance_matrix.append(l)
        self.distance_matrix = np.array(self.distance_matrix)
        print(self.distance_matrix.shape)
        
    def predict(self, example):
        
        preds = []

        for x in example:
            
            dists = []

            for y in self.X_train:
                dists.append(np.linalg.norm(x - y))
            dists = np.array(dists)

            arg = np.argsort(dists)  #Ascending Order
            top = arg[:self.k]       #Shortest K index.

            lbl = self.y_train[top]  #Labels of shortest k distances

            u, c = np.unique(lbl, return_counts=True)

            #print(dists[top])
            #print(lbl)
            #print(u[c==max(c)][0])

            preds.append(u[c==max(c)][0])

        return preds

    def get_error(self, predicted, actual):
        return sum(map(lambda x : 1 if (x[0] != x[1]) else 0, zip(predicted, actual))) / len(predicted)

    def test(self, test_input, labels):
        actual = labels
        predicted = (self.predict(test_input))
        print("error = ", self.get_error(predicted, actual))

# Add the dataset here

# Split the data 70:30 and predict.

# create a new object of class KNN

# plot a boxplot that is grouped by Species. 
# You may have to ignore the ID column

# predict the labels using KNN

# use the test function to compute the error

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

print("\nk=",5)
model = KNN(5,X_train,y_train)
model.train()
model.test(X_test, y_test)


for k in [2,3,4,6,7,8]:
    print("\nk=",k)
    model = KNN(k,X_train,y_train)
    model.test(X_test, y_test)


Data = iris.data

data_0 = Data[:,0]
data_1 = Data[:,1]
data_2 = Data[:,2]
data_3 = Data[:,3]

con = [data_0, data_1, data_2, data_3]

plt.figure(figsize = (10, 7))
plt.boxplot(con)
plt.show()