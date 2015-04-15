import sys, os, re, math, itertools, copy, csv

from os import listdir
from os.path import isfile, join
from operator import add
from random import shuffle, randint

import numpy as np
from scipy import sparse

from pyspark import SparkContext, SparkConf

TAU = 100
MIN_INIT = 0
MAX_INIT = 1

def main():
    if len(sys.argv) < 9:
        print "Not enough arguments."
        sys.exit(0)

    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])

    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])

    inputV_filepath = sys.argv[6]
    outputW_filepath = sys.argv[7]
    outputH_filepath = sys.argv[8]

    master = "local[" + sys.argv[2] + "]"
    conf = SparkConf().setAppName("10605 HW7 youngwop").setMaster(master)
    sc = SparkContext(conf=conf)

    if os.path.isdir(inputV_filepath):
        file_map = sc.wholeTextFiles(inputV_filepath)
        V_list = file_map.flatMap(map_dir)
    else:
        file_map = sc.textFile(inputV_filepath)
        V_list = file_map.map(map_file)

    V_list.cache()

    w_h_key = sc.parallelize(range(num_workers))
    tuple_key = w_h_key.flatMap(lambda x: [(x,y) for y in range(num_workers)])

    MAX_UID = V_list.map(lambda ((uid, mid), _): uid).max()
    MAX_MID = V_list.map(lambda ((uid, mid), _): mid).max()

    W_zip = sc.parallelize([(x, np.random.uniform(MIN_INIT, MAX_INIT,
        size=(((MAX_UID - x - 1) / num_workers) + 1, num_factors))) \
                for x in xrange(num_workers)])
    H_zip = sc.parallelize([(x, np.random.uniform(MIN_INIT, MAX_INIT,
        size=(num_factors, ((MAX_MID - x - 1) / num_workers) + 1))) \
                for x in xrange(num_workers)])

    def get_index(uid, mid):
        return (uid - 1) % num_workers, (mid - 1) % num_workers

    V_zip = V_list.keyBy(lambda ((uid, mid), _): get_index(uid, mid)). \
                map(lambda (x,y): (x,[y])).reduceByKey(add). \
                union(tuple_key.map(lambda x: (x, [])))

    V_zip.cache()

    V_row = V_zip.map(lambda ((row, col), res): ((row, col),
        [(uid - 1) / num_workers for ((uid, mid), rating) in list(res)])). \
        reduceByKey(add)
    V_col = V_zip.map(lambda ((row, col), res): ((row, col),
        [(mid - 1) / num_workers for ((uid, mid), rating) in list(res)])). \
        reduceByKey(add)
    V_rating = V_zip.map(lambda (key, res): (key,
        [rating for ((uid, mid), rating) in list(res)])). \
        reduceByKey(add)

    V_mat = V_row.groupWith(V_col, V_rating). \
                map(lambda ((r, c), (row, col, data)):
                    ((r, c), sparse.csr_matrix((list(data)[0], (list(row)[0], list(col)[0])),
                        shape=(((MAX_UID - r - 1) / num_workers) + 1,
                            ((MAX_MID - c - 1) / num_workers) + 1))))

    V_mat.cache()
    iter_count = 0

    for i in xrange(num_iterations):
        j_arr = range(num_workers)
        shuffle(j_arr)

        for j in j_arr:
            target_V = V_mat.filter(lambda ((x1, x2), _): x1 == ((x2 + j) % num_workers))
            target_W = W_zip.map(lambda (x, _): ((x, (x - j) % num_workers), _))
            target_H = H_zip.map(lambda (x, _): (((x + j) % num_workers, x), _))

            res = target_V.groupWith(target_W, target_H).map(lambda ((w_index, h_index), (V, W, H)): \
                    ((w_index, h_index), dsgd(list(V)[0], list(W)[0], list(H)[0], w_index, h_index, \
                        beta_value, lambda_value, iter_count))).collect()

            W_zip = sc.parallelize([(w_index, W_new) for ((w_index, h_index), (W_new, H_new, iter_count_new)) in res])
            H_zip = sc.parallelize([(h_index, H_new) for ((w_index, h_index), (W_new, H_new, iter_count_new)) in res])
            iter_count = sum([val for (_, (_, _, val)) in res])

    W_newzip = W_zip.collect()
    H_newzip = H_zip.collect()

    W_sorted = sorted(W_newzip, key=lambda x: x[0])
    H_sorted = sorted(H_newzip, key=lambda x: x[0])

    W_final = np.zeros((MAX_UID, num_factors))
    H_final = np.zeros((num_factors, MAX_MID))

    for (axis, W_loc) in W_sorted:
        for i in xrange(W_loc.shape[0]):
            W_final[num_workers * i + axis, :] = W_loc[i, :]

    for (axis, H_loc) in H_sorted:
        for i in xrange(H_loc.shape[1]):
            H_final[:, num_workers * i + axis] = H_loc[:, i]

    #print np.dot(W_final, H_final)

    W_csv = open(outputW_filepath, 'w')
    H_csv = open(outputH_filepath, 'w')

    W_write = csv.writer(W_csv, delimiter=',')
    H_write = csv.writer(H_csv, delimiter=',')

    W_write.writerows(W_final)
    H_write.writerows(H_final)

    W_csv.close()
    H_csv.close()

    return

def dsgd(V, W, H, w_index, h_index, beta_value, lambda_value, iter_count):
    L = nzsl(V, W, H, lambda_value)
    L_prev = sys.float_info.max
    V_loc = V.tocoo()
    sgd_count = 0

    data_arr = [(x,y,z) for x,y,z in itertools.izip(V_loc.row, V_loc.col, V_loc.data)]
    shuffle(data_arr)

    for uid, mid, rating in data_arr:
        W_old_row = copy.deepcopy(W[uid, :])
        H_old_col = copy.deepcopy(H[:, mid])

        epsilon_n = math.pow(TAU + iter_count + sgd_count, -beta_value)

        W[uid, :] = W_old_row - epsilon_n * \
            (-2 * (rating - np.dot(W_old_row, H_old_col)) * H_old_col + \
            2 * (lambda_value / V[uid, :].nnz) * np.transpose(W_old_row))
        H[:, mid] = H_old_col - epsilon_n * \
            (-2 * (rating - np.dot(W_old_row, H_old_col)) * np.transpose(W_old_row) + \
            2 * (lambda_value / V[:, mid].nnz) * H_old_col)

        L_prev = L
        L = nzsl(V, W, H, lambda_value)

        if L >= L_prev:
            W[uid, :] = W_old_row
            H[:, mid] = H_old_col
            return W, H, sgd_count

        sgd_count += 1

    return W, H, sgd_count

def nzsl(V, W, H, lambda_value):
    res = 0.0
    V_loc = V.tocoo()

    for uid, mid, rating in itertools.izip(V_loc.row, V_loc.col, V_loc.data):
        res += math.pow(rating - np.dot(W[uid, :], H[:, mid]), 2)

    res += lambda_value * (sum(np.add.reduce(W * W)) + sum(np.add.reduce(H * H)))

    return res

def map_dir(t):
    arr = t[1].split("\n", 1)
    mid = int(re.findall('\d+', arr[0])[0])
    tmp = [x.split(",") for x in arr[1].split("\n")]
    return [((int(elem[0]), mid), int(elem[1])) for elem in tmp if len(elem) == 3]

def map_file(t):
    arr = t.split(",")
    return ((int(arr[0]), int(arr[1])), int(arr[2]))

if __name__ == "__main__":
    main()
