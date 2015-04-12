import sys, os, re, math

from os import listdir
from os.path import isfile, join

import numpy as np
from scipy import sparse

from pyspark import SparkContext, SparkConf

MAX_UID = 2649429
MAX_MID = 17770
TAU = 100

MIN_RATING = 1
MAX_RATING = 5

def main():
    if len(sys.argv) < 9:
        print "Not enough arguments."
        sys.exit(0)

    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])

    beta_value = float(sys.argv[4])
    lambda_vale = float(sys.argv[5])

    inputV_filepath = sys.argv[6]
    outputW_filepath = sys.argv[7]
    outputH_filepath = sys.argv[8]

    master = "local[" + sys.argv[2] + "]"
    conf = SparkConf().setAppName("10605 HW7 youngwop").setMaster(master)
    sc = SparkContext(conf=conf)

    if os.path.isdir(inputV_filepath):
        file_map = sc.wholeTextFiles(inputV_filepath)
        V_list = file_map.flatMap(map_file)
    else:
        file_map = sc.textFile(inputV_filepath)

    blk_w_num = num_workers
    blk_h_num = num_workers

    w_key = sc.parallelize(range(blk_w_num))
    h_key = sc.parallelize(range(blk_h_num))
    tuple_key = w_key.flatMap(lambda x: [(x,y) for y in range(blk_h_num)])

    W_val = w_key.map(lambda x: np.random.uniform(MIN_RATING, MAX_RATING,
        size=(((MAX_UID - x - 1) / blk_w_num) + 1, num_factors)))
    H_val = h_key.map(lambda x: np.random.uniform(MIN_RATING, MAX_RATING,
        size=(num_factors, ((MAX_MID - x - 1) / blk_h_num) + 1)))

    W_zip = w_key.zip(W_val)
    H_zip = h_key.zip(H_val)

    blk_w_size = int(math.ceil(float(MAX_UID) / blk_w_num))
    blk_h_size = int(math.ceil(float(MAX_MID) / blk_h_num))

    V_zip = V_list.keyBy(lambda ((uid, mid), _): (int((uid - 1) / blk_w_size),
        int((mid - 1) / blk_h_size))).groupByKey()

    V_row = V_zip.map(lambda (key, res): (key, np.array([(uid - 1) %
        blk_w_size for ((uid, mid), rating) in list(res)])))
    V_col = V_zip.map(lambda (key, res): (key, np.array([(mid - 1) %
        blk_h_size for ((uid, mid), rating) in list(res)])))
    V_rating = V_zip.map(lambda (key, res): (key, np.array(
        [rating for ((uid, mid), rating) in list(res)])))

    V_mat_index = V_row.join(V_col)
    V_mat_zip = V_mat_index.join(V_rating)
    V_mat = V_mat_zip.map(lambda ((r, c), ((row, col), data)):
            ((r, c), sparse.csr_matrix((data, (row, col)),
                shape=(((MAX_UID - r - 1) / blk_w_num) + 1,
                    ((MAX_MID - c - 1) / blk_h_num) + 1))))

    for it_num in xrange(1, num_iterations + 1):
        for j in xrange(num_workers):
            target_V = V_mat.filter(lambda ((x1, x2), _): x1 == ((x2 + j) % num_workers))
            target_W = W_zip.map(lambda (x, _): ((x, (x - j) % num_workers), _))
            target_H = H_zip.map(lambda (x, _): (((x + j) % num_workers, x), _))
            target_W_H = target_W.join(target_H)
            V_W_H = target_V.join(target_W_H)
            res = V_W_H.map(lambda ((w_index, h_index), (V, (W, H))):
                    dsgd(V, W, H, w_index, h_index, beta_value, lambda_value,
                         blk_w_size, blk_h_size, it_num))

def dsgd(V, W, H, w_index, h_index, beta_value, lambda_value,
        blk_w_size, blk_h_size, it_num):
    V_local = [(((uid - 1) % blk_w_size, (mid - 1) % blk_h_size),
        rating) for ((uid, mid), rating) in list(V)]

    L = nzsl(V_local, W, H, lambda_value)
    L_prev = sys.float_info.max

    epsilon_n = math.pow(TAU + it_num, -beta_value)

    while L < L_prev:
        L_prev = L

        for ((uid, mid), rating) in V_local:
            #W_row_index = (uid - 1) / blk_w_size
            #H_col_index = (mid - 1) / blk_h_size

            W_new_row = W[uid, :] - epsilon_n * (-2 * (rating - np.dot(W[uid, :], H[:, mid])[0]) * H[:, mid] + 2 * (lambda_val / N) * np.transpose(W[uid, :]))

    return W, H

def nzsl(V, W, H, lambda_value):
    res = 0.0

    for ((uid, mid), rating) in V:
        res += math.pow(rating - np.dot(W[uid, :], H[:, mid])[0], 2)

    res += lambda_value * (sum(np.add.reduce(W * W)) + sum(np.add.reduce(H * H)))

    return res

def map_file(t):
    arr = t[1].split("\n", 1)
    mid = int(re.findall('\d+', arr[0])[0])
    tmp = [x.split(",") for x in arr[1].split("\n")]
    return [((int(elem[0]), mid), int(elem[1])) for elem in tmp if len(elem) == 3]

if __name__ == "__main__":
    main()
