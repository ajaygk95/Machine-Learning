import numpy as np
import argparse


class Perceptron:
    def __init__(self):
        # Initialize weights, bias to 0
        self.weights = []
        self.bias = 0
        self.iterations_run = 0
        self.max_iterations = 15000

    def fit(self, input_x, output, epsilon, bias_rate):

        # init weight, bias to 0
        self.weights = [0.0 for _ in range(len(input_x[0]))]
        self.bias = 0

        input_len = len(input_x)
        epoch = 0
        error = epsilon + 1

        # Loop till error is less than given error or number of iterations is exhausted
        while epoch < self.max_iterations and error > epsilon:
            record_len = 0
            epoch_err = 0
            while record_len < input_len:
                x = input_x[record_len]
                y = output[record_len]
                record_len += 1

                # Calculate <X> * <w>
                prod_wx = self.dot_prod(x)
                # if wx > 0, then 1 else 0
                h_of_x = self.sign(prod_wx)
                # Calculate the number of errors i.e h(x) != y
                if h_of_x != y:
                    # err is +- 1
                    err = y - h_of_x
                    # Increase or decrease bias
                    self.bias += err * bias_rate
                    # new w = w +/- x
                    self.weights += np.multiply(x, err)
                    epoch_err += 1
            epoch += 1
            error = epoch_err / input_len
        self.iterations_run = epoch

    def dot_prod(self, input_x):
        return np.dot(input_x, self.weights) + self.bias

    def sign(self, wx):
        return np.where(wx > 0.0, 1, 0)

    def predict(self, X):
        # Calculate <X> * <w>
        prod_wx = self.dot_prod(X)
        # if wx > 0, then 1 else 0
        sign_y = self.sign(prod_wx)
        return sign_y

    def erm(self, input_X, output_Y):
        err = 0
        for idx, i in enumerate(self.predict(input_X)):
            if i != output_Y[idx]:
                err += 1
        return err / len(input_X)


def main():
    # Parse all arguments from cmd line
    # Read data into x,y list
    # Shuffle the data
    # 1. ERM 2. Kfold
    inputs = parse_arguments()
    file_path = inputs.file_path
    mode = inputs.mode
    bias_factor = inputs.b

    epsilon = inputs.e

    k_fold = inputs.k_fold

    is_shuffle = inputs.is_shuffle

    # Read input as list of lists and output as list and shuffle data
    dataset_input, dataset_output = read_data(file_path)

    if is_shuffle:
        # Ref: https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
        dataset_input, dataset_output = shuffle_data(dataset_input, dataset_output)

    # Check mode if erm or kfold
    if mode == 'erm':
        perceptron = Perceptron()
        # Calculate the weights and bias. In case of linearly separable error will be 0
        perceptron.fit(dataset_input, dataset_output, epsilon, bias_factor)
        # Calculate empirical risk on the entire dataset
        empirical_error = perceptron.erm(dataset_input, dataset_output)

        print("Perceptron ERM output")
        print("\titerations: {0}".format(perceptron.iterations_run))
        print("\tbias and weights: {0} {1}".format(perceptron.bias, perceptron.weights))
        print("\tempirical_error: {0}".format(empirical_error))

    elif mode == 'kfold':
        # Get bias,weights, error and number of iterations for each fold
        kfold_bias_weights, kfold_err, kfold_iter = kfold_validation(dataset_input, dataset_output, k_fold, epsilon,
                                                                     bias_factor)
        i = 0
        while i < len(kfold_bias_weights):
            print("kfold-{0}".format(i + 1))
            print("\titerations: {0}".format(kfold_iter[i]))
            print("\tbias and weights: {0}".format(kfold_bias_weights[i]))
            print("\tValidation error: {0}".format(kfold_err[i]))
            i += 1
        print("\nMean validation error over all kfolds: {0}\n".format(np.mean(kfold_err)))

    else:
        print('Mode not supported')


def parse_arguments():
    # Parse all input args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="path to data.", action="store", dest="file_path",
                        default="linearly-separable-dataset.csv", type=str)
    parser.add_argument("-m", "--mode", help="erm/kfold", action="store", dest="mode",
                        default="erm", type=str)

    parser.add_argument("-e", "--epsilon", help="Epsilon for perceptron. Default = 0",
                        action="store",
                        dest="e",
                        default=0.00, type=float)

    parser.add_argument("-b", "--bias_factor",
                        help="Multiplication factor used in updating the new bias value. Default = 1",
                        action="store",
                        dest="b",
                        default=1, type=float)

    parser.add_argument("-k", "--kfold", help="number of folds", action="store",
                        dest="k_fold",
                        default=10, type=int)

    parser.add_argument("-s", "--shuffle", help="shuffle the dataset. Default = False", action="store_true",
                        dest="is_shuffle",
                        default=False)
    return parser.parse_args()


def read_data(file_path, skip_header=True):
    # Read X and Y into separate lists
    x = []
    y = []
    with open(file_path, 'r') as reader:
        if skip_header:
            next(reader)
        for line in reader:
            line = line.rstrip('\n')
            data = line.split(',')
            data = [float(i) for i in data]
            x.append(data[:-1])
            y.append(data[-1])
    return x, y


def shuffle_data(dataset_input, dataset_output):
    # zip x,y shuffle them and unzip x,y
    zip_list = list(zip(dataset_input, dataset_output))
    np.random.shuffle(zip_list)
    return zip(*zip_list)


def kfold_validation(input_data, output_data, k_fold, epsilon, bias_factor):
    # Split data equally into n folds
    kfold_input = np.array_split(input_data, k_fold)
    kfold_output = np.array_split(output_data, k_fold)

    kfold_err = []
    kfold_bias_weights = []
    kfold_iter = []
    perceptron = Perceptron()

    for k in range(k_fold):
        x_k = []
        y_k = []
        x_test = []
        y_test = []
        idx = 0
        # Split train and validation data
        while idx < len(kfold_input):
            if idx != k:
                x_k += list(kfold_input[idx])
                y_k += list(kfold_output[idx])
            else:
                x_test = list(kfold_input[idx])
                y_test = list(kfold_output[idx])
            idx += 1
        # Call fit on train data
        perceptron.fit(x_k, y_k, epsilon, bias_factor)
        kfold_bias_weights.append([perceptron.bias] + list(perceptron.weights))
        # Calculate erm on validation set
        kfold_err.append(perceptron.erm(x_test, y_test))
        kfold_iter.append(perceptron.iterations_run)
    return kfold_bias_weights, kfold_err, kfold_iter


if __name__ == "__main__":
    main()
