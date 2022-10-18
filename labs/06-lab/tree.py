import numpy as np

class Node:
    def __init__(self, attr, vote):
        self.attribute = attr
        self.left = None
        self.right = None
        self.vote = vote
    
    def print_node_at(self, depth):
        if depth == 0:
            self.print_node_at(depth + 1)
        else:
            pretty_print = ""
            for _ in range(depth):
                pretty_print += "| "
            pretty_print += self.attribute + ' = 0: '
            print(pretty_print)
            if self.left is not None:
                self.left.print_node_at(depth + 1)
        
            pretty_print = ""
            for _ in range(depth):
                pretty_print += "| "
            pretty_print += self.attribute + ' = 1: '
            print(pretty_print)
            if self.right is not None:
                self.right.print_node_at(depth + 1)
        return


# (a) the left and right children of a node are denoted as
# node.left and node.right respectively, each is of type Node
# 
# (b) the attribute for a node is denoted as node.attribute and has
# type str
# 
# (c) if the node is a leaf, then node.vote of type str holds the
# prediction from the majority vote; if node is an internal
# node, then node.vote has value None
# 
# (d) assume all attributes have values 0 and 1 only; further
# assume that the left child corresponds to an attribute value
# of 1, and the right child to a value of 0

def get_prediction(node, example):
    # example is a dictionary which holds the attributes and the
    # values of the attribute (ex. example[’X’] = 0)
    if node.left is None and node.right is None:
        # leaf node
        return node.vote
    else:
        # your code here
        return


class SimpleDecisionTree:

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None
        self.default_class = None
        self.target_name = None


    def read(self, dataset_path):
        return np.genfromtxt(fname=dataset_path, delimiter = '\t', names = True)


    def get_target_name(self, data):
        return data.dtype.names[-1]


    def gini_split(self, data, attr):
        # compute the gini of the split on attr
        pass
    

    def get_majority_vote(self, subset):
        # get the majority vote from a subset of the dataset
        pass


    def is_pure(data, target_name):
        # returns true if all data has the same target value
        pass

    def get_best_attribute(self, data):
        # returns None if none of the attributes are good
        if self.is_pure(data, self.target_name):
            return None
        
        pass

    
    def get_subset(self, data, attr):
        left = ... # return the rows of the dataset where attribute == 1
        right = ... # return the rows of the dataset where attribute == 0
        return left, right


    def get_node(self, data, max_depth):
        attr = self.get_best_attribute(data)
        if attr is None:
            return None

        node = Node(attr, None)

        if max_depth == 0:
            if data is None:
                node.vote = self.default_class
            else:
                node.vote = self.get_majority_vote(data, self.target_name)
            return node

        left, right = self.get_subset(data, node.attribute)

        node.left = self.get_node(left, max_depth - 1)
        node.right = self.get_node(right, max_depth - 1)

        if node.left is None and node.right is None:
            # cast a vote
            node.vote = self.get_majority_vote(data)

        return node


    def train(self, dataset):
        data = self.read(dataset)
        self.default_class = self.get_majority_vote(data)
        self.target_name = self.get_target_name(data)
        self.tree = self.get_node(data, self.max_depth)

    
    def predict(self, example):
        return get_prediction(self.tree, example)


    def get_error(self, predicted, actual):
        return sum(map(lambda x : 1 if (x[0] != x[1]) else 0, zip(predicted, actual))) / len(predicted)


    def test(self, test_input):
        predicted = []
        data = self.read(test_input)
        target= self.get_target_name(data)
        actual = data[target]
        for example in data:
            print(self.predict(example))
            predicted.append(self.predict(example))
        print("error = ", self.get_error(predicted, actual))
    

    def pretty_print(self):
        self.tree.print_node_at(0)
    

my_decision_tree = SimpleDecisionTree(2)
my_decision_tree.train('education_train.tsv')
my_decision_tree.pretty_print()
my_decision_tree.test("education_test.tsv")
