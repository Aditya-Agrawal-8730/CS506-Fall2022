class KNN:

    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.distance_matrix = None
    
    def train(self):
        self.distance_matrix = ...

    def predict(self, example):
        return ...

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
