import numpy as np
import argparse
import matplotlib.pyplot as plt


class AdaBoost:
    def __init__(self):
        # Initialize weights, decision stumps
        self.weights = []
        self.errors = []
        self.decision_stump_hypo = []

    def fit(self, input_x, output_y, T):
        # init weight, error, decision stump to 0
        m = len(input_x)
        self.weights = [0.0 for _ in range(T)]
        self.errors = [0 for _ in range(T)]
        self.decision_stump_hypo = [(0, 0.0) for _ in range(T)]

        # init D vector to 1/m
        D = [1 / m for _ in range(m)]
        for t in range(T):
            # Calculate best decision stump for t^th iteration based on erm for decision stump
            j, theta = self.erm_decision_stump(input_x, output_y, D)
            self.decision_stump_hypo[t] = j, theta
            # Predict y and calculate the error
            ht_x = self.perdict(input_x, self.decision_stump_hypo[t])
            self.errors[t] = self.epsilon(ht_x, output_y, D)
            # If no error then break the loop
            if self.errors[t] == 0:
                break
            # Update weights for t^th iteration
            self.weights[t] = np.log((1 / self.errors[t]) - 1) / 2

            # Update the value of D
            exp_w_y_h = np.exp(np.multiply(np.multiply(ht_x, output_y), -1 * self.weights[t])).tolist()
            newD = np.multiply(D, exp_w_y_h).tolist()
            normalizer = np.dot(exp_w_y_h, D)
            D = newD / normalizer

    def epsilon(self, y_hat, output_y, D):
        # if h(x) != y then error += D_i
        i = 0
        error = 0
        while i < len(output_y):
            if y_hat[i] != output_y[i]:
                error += D[i]
            i += 1
        return error

    def perdict(self, input_x, hypothesis):
        # Based on the given hypothesis. If x_j > theta then 1 else -1
        j, theta = hypothesis
        x_j = []
        for record in input_x:
            x_j.append(record[j])
        return np.where(np.array(x_j, copy=False) > theta, 1.0, -1.0)

    def erm_decision_stump(self, input_x, output_y, D):
        # erm on decision stumps. Return j_start, theta_star
        f_star = np.infty
        theta_star = 0.0
        j_star = 0

        for j in range(len(input_x[0])):
            x_j = []
            # Collect all x_j in a list
            for record in input_x:
                x_j.append(record[j])

            # x_j, y, D --> list and sort
            zip_list = list(zip(x_j, output_y, D))
            sorted_list = sorted(zip_list, key=lambda x: x[0])

            # sum all D_i where y = 1
            f = 0
            for rec in sorted_list:
                if rec[1] == 1:
                    f += rec[2]

            if f < f_star:
                f_star = f
                theta_star = sorted_list[0][0] - 1
                j_star = j

            i = 0
            while i < len(sorted_list) - 1:
                rec = sorted_list[i]
                next_rec = sorted_list[i + 1]
                i += 1
                f -= (rec[1] * rec[2])
                if f < f_star and (rec[0] != next_rec[0]):
                    f_star = f
                    theta_star = (rec[0] + next_rec[0]) / 2
                    j_star = j

        return j_star, theta_star

    def erm(self, input_x, output_y, weights, hypothesis):
        # hypothesis is a list of all weak learners having j_start, theta_star
        # weights is also a vector of w^t
        t = 0
        predictions = []
        while t < len(hypothesis):
            ht = hypothesis[t]
            wt = weights[t]
            # predict y based on hypo^t
            ht_x = self.perdict(input_x, ht)
            # w^t * h^t(x). Scalar multiplication of w^t with the list of predicted values.
            predictions.append(np.multiply(ht_x, wt))
            t += 1
        # Add all corresponding column values in the prediction list using axis=0 and give the actual prediction of y.
        # this is sum_{1-t}{w^t * h^t(x)}
        hs_x = np.where(np.sum(predictions, axis=0) > 0, 1, -1)
        # Calculate the errors
        i = 0
        error = 0
        while i < len(output_y):
            if hs_x[i] != output_y[i]:
                error += 1
            i += 1
        return error / len(output_y)


def main():
    # Parse all arguments from cmd line
    inputs = parse_arguments()
    file_path = inputs.file_path
    mode = inputs.mode
    T = inputs.t

    k_fold = inputs.k_fold
    is_plot = inputs.isPlot
    is_shuffle = inputs.is_shuffle

    # Read input as list of lists and output as list and shuffle data
    dataset_input, dataset_output = read_data(file_path)

    # If y == 0 then change it to -1
    y = 0
    while y < len(dataset_output):
        if dataset_output[y] == 0:
            dataset_output[y] = -1
        y += 1

    if is_shuffle:
        # Ref: https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
        dataset_input, dataset_output = shuffle_data(dataset_input, dataset_output)

    if mode == 'erm':
        # Call fit and calculate the erm on the entire dataset
        adaBoost = AdaBoost()
        adaBoost.fit(dataset_input, dataset_output, T)

        for t in range(T):
            print("AdaBoost-{0}".format(t + 1))
            print("\tweights: {0}".format(adaBoost.weights[t]))
            print("\tDecision-Stump: {0}".format(adaBoost.decision_stump_hypo[t]))
            print("\tempirical_error: {0}".format(adaBoost.errors[t]))
        # Calculate empirical risk on the entire dataset
        error = adaBoost.erm(dataset_input, dataset_output, adaBoost.weights, adaBoost.decision_stump_hypo)
        print("\n Total error after {0} boosting is: {1}".format(T, error))

    elif mode == 'kfold':
        # If they want to plot, then run kfold from t=1 to T.
        if is_plot:
            mean_kfold_validation_err = []
            mean_kfold_erm_err = []
            t_array = []
            for t in range(1, T + 1):
                print("Boosters: {0}".format(t))
                validation_err, erm_err = kfold_runner(dataset_input, dataset_output, k_fold, t)
                mean_kfold_validation_err.append(validation_err)
                mean_kfold_erm_err.append(erm_err)
                t_array.append(t)

            fig, ax = plt.subplots(1, figsize=(15, 15))
            ax.plot(t_array, mean_kfold_validation_err, label="validation_error")
            ax.plot(t_array, mean_kfold_erm_err, label="erm_error")
            ax.set_title("Validation error v/s ERM error", fontsize=17)
            ax.set_xlabel('Number of Boosters', fontsize=15)
            ax.legend()
            plt.show()

        else:
            print("Boosters: {0}".format(T))
            kfold_runner(dataset_input, dataset_output, k_fold, T)

    else:
        print('Mode not supported')


def parse_arguments():
    # Parse all input args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="path to data.", action="store", dest="file_path",
                        default="Breast_cancer_data.csv", type=str)
    parser.add_argument("-m", "--mode", help="erm/kfold", action="store", dest="mode",
                        default="erm", type=str)

    parser.add_argument("-t", "--trounds", help="Number of boosting rounds. Default=10", action="store", dest="t",
                        default=10, type=int)

    parser.add_argument("-k", "--kfold", help="number of folds. Default=10", action="store",
                        dest="k_fold",
                        default=10, type=int)
    parser.add_argument("-p", "--plot", help="Plot the validation v/s erm error plot for T boosters. Default = False",
                        action="store_true",
                        dest="isPlot",
                        default=False)
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


def kfold_runner(dataset_input, dataset_output, k_fold, T):
    kfold_bias_weights, kfold_hypothesis, kfold_validation_err, kfold_erm_error = kfold_validation(dataset_input,
                                                                                                   dataset_output,
                                                                                                   k_fold, T)
    i = 0
    while i < len(kfold_bias_weights):
        print("kfold-{0}".format(i + 1))
        print("\tweights: {0}".format(kfold_bias_weights[i]))
        print("\tDecision-Stump: {0}".format(kfold_hypothesis[i]))
        print("\terror: {0}".format(kfold_validation_err[i]))
        i += 1
    mean_kfold_validation_err = np.mean(kfold_validation_err)
    mean_kfold_erm_err = np.mean(kfold_erm_error)
    print("\nMean Validation error over all kfolds: {0}".format(mean_kfold_validation_err))
    print("\nMean ERM error over all kfolds: {0}\n".format(mean_kfold_erm_err))
    return mean_kfold_validation_err, mean_kfold_erm_err


def kfold_validation(input_data, output_data, k_fold, T):
    # Split data equally into n folds
    kfold_input = np.array_split(input_data, k_fold)
    kfold_output = np.array_split(output_data, k_fold)

    kfold_validation_err = []
    kfold_weights = []
    kfold_hypothesis = []
    kfold_erm_error = []

    adaBoost = AdaBoost()

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
        adaBoost.fit(x_k, y_k, T)
        kfold_weights.append(adaBoost.weights)
        kfold_hypothesis.append(adaBoost.decision_stump_hypo)

        # Calculate erm on validation set
        kfold_validation_err.append(adaBoost.erm(x_test, y_test, adaBoost.weights, adaBoost.decision_stump_hypo))
        kfold_erm_error.append(adaBoost.erm(x_k, y_k, adaBoost.weights, adaBoost.decision_stump_hypo))

    return kfold_weights, kfold_hypothesis, kfold_validation_err, kfold_erm_error


if __name__ == "__main__":
    main()
