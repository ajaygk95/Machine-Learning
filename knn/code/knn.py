import numpy as np
import argparse


class Knn:
    def __init__(self):
        self.k = 1
        self.input_x = []
        self.output = []

    def fit(self, input_x, output, k):
        self.input_x = input_x
        self.output = output
        self.k = k

    def predict(self, X):
        y_pred = []
        x_t = len(X)
        record_len = 0
        while record_len < x_t:
            dist_xt_xm = []
            x_test = X[record_len]
            record_len += 1
            for idx, x in enumerate(self.input_x):
                euclid_dist = self.calc_distance(x_test, x)
                dist_xt_xm.append((euclid_dist, self.output[idx]))

            dist_xt_xm.sort(key=lambda x: x[0])
            k_list = []
            for i in range(self.k):
                k_list.append(dist_xt_xm[i][1])
            count_1 = k_list.count(1)
            if count_1 > (len(k_list) // 2):
                y_pred.append(1)
            else:
                y_pred.append(0)

        return y_pred

    def calc_distance(self, x1, x2):
        dist = 0
        for i in range(len(x1)):
            dist += np.square(x1[i] - x2[i])
        return np.sqrt(dist)


def main():
    inputs = parse_arguments()
    file_path = inputs.file_path
    k = inputs.k_n
    is_shuffle = inputs.is_shuffle
    train_per = inputs.train_size

    dataset_input, dataset_output = read_data(file_path)

    if is_shuffle:
        # Ref: https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
        dataset_input, dataset_output = shuffle_data(dataset_input, dataset_output)

    m = len(dataset_input)
    train_size = int(train_per * m)

    X_train = dataset_input[0:train_size]
    Y_train = dataset_output[0:train_size]
    X_test = dataset_input[train_size:]
    Y_test = dataset_output[train_size:]

    print("Train Dataset size: ", len(X_train))
    print("Test Dataset size: ", len(X_test))
    print("K-Neighbour: ", k)

    knn = Knn()
    knn.fit(X_train, Y_train, k)
    y_pred = knn.predict(X_test)

    err = 0
    for i in range(len(Y_test)):
        if y_pred[i] != Y_test[i]:
            err += 1

    accuracy = (len(Y_test) - err) / len(Y_test)

    print("No of miss-classifications: ", err)
    print("Accuracy %: ", accuracy * 100)


def parse_arguments():
    # Parse all input args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="path to data.", action="store", dest="file_path",
                        default="Breast_cancer_data.csv", type=str)
    parser.add_argument("-k", "--knn", help="K-Neighbors", action="store", dest="k_n", default=3, type=int)
    parser.add_argument("-t", "--train_size", help="Train-split percentage, 0-1", action="store", dest="train_size",
                        default=0.8,
                        type=float)
    parser.add_argument("-s", "--shuffle", help="shuffle the dataset. Default = False", action="store_true",
                        dest="is_shuffle", default=False)

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


if __name__ == "__main__":
    main()
