import numpy as np
import matplotlib.pyplot as plt

n_samples = 500
C = np.array([[0.1, 0.6], [2., .6]])
X = np.random.randn(n_samples, 2) @ C + np.array([-6, 3])
plt.axis('equal')
plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.8)
plt.title("Raw Data")
plt.show()

Xc = X - np.mean(X,axis=0)
plt.axis('equal')
plt.scatter(Xc[:, 0], Xc[:, 1], s=10, alpha=0.8, color='r')
plt.title("Mean Centered Data")
plt.show()

# def pca(X):
#   # Data matrix X, assumes 0-centered
#   n, m = X.shape
#   assert np.allclose(X.mean(axis=0), np.zeros(m))
#   # Compute covariance matrix
#   C = np.dot(X.T, X) / (n-1)
#   # Eigen decomposition
#   eigen_vals, eigen_vecs = np.linalg.eig(C)
#   # Project X onto PC space
#   X_pca = np.dot(X, eigen_vecs)
#   return X_pca

# X_pca = pca(Xc.copy())
# plt.axis('equal')
# plt.scatter(Xc[:,0],Xc[:,1], color='r')
# plt.scatter(X_pca[:,0], X_pca[:,1])
# plt.show()

u, s, vt = np.linalg.svd(Xc,full_matrices=False)
plt.plot(range(1,len(s)+1),s)
plt.title("Singular Values")
plt.show()

# show ouput from svd is the same
shiftedX = u
plt.axis('equal')
plt.scatter(Xc[:,0],Xc[:,1], color='r')
plt.scatter(shiftedX[:,0], shiftedX[:,1])
endpoints = np.array([[-10],[10]]) @ vt[[0],:]
_ = plt.plot(endpoints[:,0], endpoints[:,1], 'g-')
plt.show()

# project onto one of the singular vectors
scopy = s.copy()
scopy[1] = 0.0
reducedX = u @ np.diag(scopy) @ vt
plt.axis('equal')
plt.scatter(Xc[:,0],Xc[:,1], color='r')
plt.scatter(reducedX[:,0], reducedX[:,1])
endpoints = np.array([[-10],[10]]) @ vt[[0],:]
_ = plt.plot(endpoints[:,0], endpoints[:,1], 'g-')
plt.show()
