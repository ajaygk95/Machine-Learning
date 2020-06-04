import numpy as np
import argparse


class Kmeans:
    def __init__(self, epsilon=0.01):
        self.input_x = []
        self.distance_function = euclidean_distance
        self.k = 2
        self.epsilon = epsilon
        self.mu = []
        self.cluster_partiton = []

    def set_init_centroids(self, input_x):
        prev_i = 0
        i = prev_i
        m = len(input_x)
        for _ in range(self.k):
            while i == prev_i:
                i = np.random.randint(m)
            prev_i = i
            self.mu.append(input_x[i])

    def fit(self, input_x, distance_function, k):
        self.distance_function = distance_function
        self.k = k
        self.set_init_centroids(input_x)
        prev_mu = []
        feature_length = len(input_x[0])
        for _ in range(self.k):
            self.cluster_partiton.append({})
            prev_mu.append([0] * feature_length)

        err = self.epsilon + 1
        prev_centroid_ids = [-1 for _ in range(len(input_x))]
        while err > self.epsilon:
            for i in range(len(input_x)):
                old_dist = float('inf')
                prev_centroid_id = prev_centroid_ids[i]
                for c in range(len(self.mu)):
                    dist = distance_function(self.mu[c], input_x[i])
                    if dist < old_dist:
                        old_dist = dist
                        if c != prev_centroid_id:
                            if prev_centroid_id != -1:
                                self.cluster_partiton[prev_centroid_id].pop(i, None)

                            self.cluster_partiton[c][i] = input_x[i]
                            prev_centroid_id = c
                            prev_centroid_ids[i] = prev_centroid_id

            err = 0
            for c in range(len(self.mu)):
                cluster_dict = self.cluster_partiton[c]
                x_in_cluster = list(cluster_dict.values())
                if len(x_in_cluster) != 0:
                    self.mu[c] = np.sum(x_in_cluster, axis=0) / len(x_in_cluster)
                err += euclidean_distance(self.mu[c], prev_mu[c])
                prev_mu[c] = self.mu[c].copy()

    def predict(self, input_x):
        output_centroid = []
        for x in input_x:
            old_dist = float('inf')
            for c in range(len(self.mu)):
                dist = self.distance_function(self.mu[c], x)
                if dist < old_dist:
                    old_dist = dist
                    cluster = c
            output_centroid.append(cluster)
        return output_centroid


def main():
    inputs = parse_arguments()
    file_path = inputs.file_path
    k = inputs.k_n
    dist = inputs.dist
    e = inputs.e

    dataset_input, dataset_output = read_data(file_path)

    if dist == "euclidean":
        distance_function = euclidean_distance
    elif dist == "manhattan":
        distance_function = manhattan_distance
    else:
        print("Unknown Distance function: ", dist)
        return 1

    kmeans = Kmeans(e)
    kmeans.fit(dataset_input, distance_function, k)

    print("K-Means : ", k)
    print("Distance Function : ", dist)
    print("Centroids : ", kmeans.mu)

    output_y = check_cluster_performance(kmeans, dataset_output)
    print("Cluster-Count")
    for i in range(len(output_y)):
        print("\t Cluster-{} : {}".format(i, output_y[i]))
        total = sum(output_y[i].values())
        p0 = (output_y[i]['Count-0'] / total)*100
        p1 = (output_y[i]['Count-1'] / total)*100
        print("\t\t Y=0 % : {}".format(p0))
        print("\t\t Y=1 % : {}".format(p1))


def parse_arguments():
    # Parse all input args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="path to data.", action="store", dest="file_path",
                        default="Breast_cancer_data.csv", type=str)
    parser.add_argument("-k", "--kcluster", help="K-Clusters", action="store", dest="k_n", default=2, type=int)
    parser.add_argument("--distance", help="Distance Function. euclidean/manhattan  Default euclidean", action="store",
                        dest="dist", default="euclidean", type=str)
    parser.add_argument("-e", "--epsilon", help="Epsilon for Change in Centroid. Default = 0.0001",
                        action="store", dest="e", default=0.0001, type=float)
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


def euclidean_distance(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += np.square(x1[i] - x2[i])
    return np.sqrt(dist)


def manhattan_distance(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += np.abs(x1[i] - x2[i])
    return dist


def check_cluster_performance(kmeans, dataset_output):
    cluster_partiton = kmeans.cluster_partiton
    output_y = []
    for cluster in cluster_partiton:
        count_1 = 0
        count_0 = 0
        for key in cluster.keys():
            y = dataset_output[key]
            if y == 1:
                count_1 += 1
            else:
                count_0 += 1
        output_y.append({'Count-0': count_0, 'Count-1': count_1})
    return output_y


if __name__ == "__main__":
    main()
