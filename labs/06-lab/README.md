# Decision Trees

In this lab we will build a simple decision tree from scratch. It is simple because it can only handle attributes whose values are either 0 or 1.

## Warm up

Finish the implementation of the `get_prediction(node, example)` function.

## Lab 6

The goal is to predict whether students will get an A or not given a set of assignment grades provided. The attributes are student grades on 5 multiple choice assignments M1 through M5, 4 programming assignments P1 through P4, and the final exam F. Values of 1 indicate that a student received an A, and 0 indicates that the student did not receive an A.

Implement the following methods of the `SimpleDecisionTree` class:

- gini_split(self, dataset, attr)
- get_majority_vote(self, subset)
- get_best_attribute(self, dataset)
- is_pure(self, data)
- get_subset(self, data, attr)

Test your implementation on the testing dataset provided.

Also verify your implementation by printing the decision tree. The output of `pretty_print` for `max_depth` of 2 on the training set provided should be:

```
| F = 0: 
| | M2 = 0: 
| | | M4 = 0: 
| | | M4 = 1: 
| | M2 = 1: 
| | | M4 = 0: 
| | | M4 = 1: 
| F = 1: 
| | M4 = 0: 
| | | M2 = 0: 
| | | M2 = 1: 
| | M4 = 1: 
| | | P1 = 0: 
| | | P1 = 1: 
```
