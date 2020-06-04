import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, max_iterations, lambda_param=0.25):
        # weights include the bias param
        self.weights = []
        self.weight = []
        self.T = max_iterations
        self.eta = 1
        self.lambda_param = lambda_param

    def train(self, input_x, output_y):
        self.weights = [np.zeros(input_x.shape[1])]
        self.weight = [np.zeros(input_x.shape[1])]
        m = input_x.shape[0]
        prev_y = output_y[0]
        for t in range(self.T):
            self.eta = 1 / (1 + t)
            self.weight = self.weights[-1].copy()
            # Sample a random value from the training data
            z = np.random.randint(0, m)
            # Give alternating values to the algorithm. If the old value of y=1 train with -1 for this time
            # And vice versa
            while output_y[z] == prev_y:
                z = (z + 1) % m
            x = input_x[z]
            y = output_y[z]
            prev_y = y
            # Update weights and bias
            if y * np.dot(x, self.weight) < 1:
                self.weight[1:] -= self.eta * ((2 * self.lambda_param * self.weight[1:]) - np.dot(x[1:], y))
                self.weight[0] -= self.eta * y
            else:
                self.weight[1:] -= self.eta * 2 * self.lambda_param * self.weight[1:]
            self.weights.append(self.weight)

    def test(self, X):
        # use last weight
        prod_wx = np.dot(X, self.weight)
        return np.where(prod_wx > 0, 1, -1)


def main():
    X0, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=10)
    X1 = np.c_[np.ones((X0.shape[0])), X0]  # add one to the x-values to incorporate bias

    # change the label to -1 and +1
    y = np.where(y < 1, -1, y)

    X_train = X1[:80]
    Y_train = y[:80]
    X_test = X1[80:]
    Y_test = y[80:]

    svm = SVM(1000, 0.3)
    svm.train(X_train, Y_train)
    y_pred = svm.test(X_test)

    err = 0
    for i in range(len(Y_test)):
        if y_pred[i] != Y_test[i]:
            err += 1

    print("Bias and Weights are: ", svm.weight)
    print("No of miss-classifications: ", err)

    draw(X0, y, svm.weight)


def draw(input_x, y, weights):
    plt.figure(figsize=(15, 8))
    plt.grid(True)
    positive_x = []
    negative_x = []
    for i, label in enumerate(y):
        if label == 1:
            positive_x.append(input_x[i])
        else:
            negative_x.append(input_x[i])

    positive_x = np.array(positive_x)
    negative_x = np.array(negative_x)
    plt.plot(positive_x[:, 0], positive_x[:, 1], 'co', alpha=0.8, label="+1")
    plt.plot(negative_x[:, 0], negative_x[:, 1], 'ro', alpha=0.8, label="-1")
    plt.legend()

    x_points = []
    hyperplane = []
    decision_boundary1 = []
    decision_boundary2 = []
    slope = -(weights[1] / weights[2])
    intercept = -(weights[0]) / weights[2]
    l2_norm = 1 / np.linalg.norm(weights[1:])
    for i in np.linspace(np.amin(input_x[:, :1]), np.amax(input_x[:, :1])):
        y_ = (slope * i) + intercept
        y1 = (slope * i) + intercept + l2_norm
        y2 = (slope * i) + intercept - l2_norm
        x_points.append(i)
        hyperplane.append(y_)
        decision_boundary1.append(y1)
        decision_boundary2.append(y2)
    plt.plot(x_points, hyperplane, marker='o', color='k')
    plt.plot(x_points, decision_boundary1, x_points, decision_boundary2, color='orange', linestyle='--', linewidth=2)
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    plt.title('SVM Decision Boundary Plot', fontdict={'fontsize': 15})
    plt.show()


if __name__ == "__main__":
    main()
