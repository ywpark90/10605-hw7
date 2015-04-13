import sys, os, re, math, itertools, copy

from os import listdir
from os.path import isfile, join
from operator import add

import numpy as np
from scipy import sparse

from pyspark import SparkContext, SparkConf

#MAX_UID = 2649429
#MAX_MID = 17770
TAU = 100
MAX_UID = 8
MAX_MID = 10

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
    lambda_value = float(sys.argv[5])

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

    W_val = sc.parallelize([np.random.uniform(MIN_RATING, MAX_RATING,
        size=(((MAX_UID - x - 1) / blk_w_num) + 1, num_factors)) \
                for x in xrange(blk_w_num)])
    H_val = sc.parallelize([np.random.uniform(MIN_RATING, MAX_RATING,
        size=(num_factors, ((MAX_MID - x - 1) / blk_h_num) + 1)) \
                for x in xrange(blk_h_num)])

    W_zip = w_key.zip(W_val)
    H_zip = h_key.zip(H_val)

    blk_w_size = int(math.ceil(float(MAX_UID) / blk_w_num))
    blk_h_size = int(math.ceil(float(MAX_MID) / blk_h_num))

    blk_w_arr = [(MAX_UID - r - 1) / blk_w_num + 1 for r in xrange(blk_w_num)]
    blk_h_arr = [(MAX_MID - c - 1) / blk_h_num + 1 for c in xrange(blk_h_num)]

    blk_w_arr = [sum(blk_w_arr[:i]) for i in xrange(blk_w_num)]
    blk_h_arr = [sum(blk_h_arr[:i]) for i in xrange(blk_h_num)]

    blk_w_rem = MAX_UID % blk_w_num
    blk_h_rem = MAX_MID % blk_h_num

    blk_w_cutoff = blk_w_size * blk_w_rem
    blk_h_cutoff = blk_h_size * blk_h_rem

    def get_index(uid, mid):
        if blk_w_cutoff == 0 or uid <= blk_w_cutoff:
            row = (uid - 1) / blk_w_size
        else:
            row = ((uid - blk_w_cutoff - 1) / (blk_w_size - 1)) + blk_w_rem

        if blk_h_cutoff == 0 or mid <= blk_h_cutoff:
            col = (mid - 1) / blk_h_size
        else:
            col = ((mid - blk_h_cutoff - 1) / (blk_h_size - 1)) + blk_h_rem

        return row, col

    V_zip = V_list.keyBy(lambda ((uid, mid), _): get_index(uid, mid)).groupByKey()
    V_empty = tuple_key.map(lambda x: (x, []))

    V_row = V_zip.map(lambda ((row, col), res): ((row, col),
        [uid - blk_w_arr[row] - 1 for ((uid, mid), rating) in list(res)]))
    V_col = V_zip.map(lambda ((row, col), res): ((row, col),
        [mid - blk_h_arr[col] - 1 for ((uid, mid), rating) in list(res)]))
    V_rating = V_zip.map(lambda (key, res): (key,
        [rating for ((uid, mid), rating) in list(res)]))

    V_row = V_row.union(V_empty).reduceByKey(add)
    V_col = V_col.union(V_empty).reduceByKey(add)
    V_rating = V_rating.union(V_empty).reduceByKey(add)

    V_mat_index = V_row.join(V_col)
    V_mat_zip = V_mat_index.join(V_rating)
    V_mat = V_mat_zip.map(lambda ((r, c), ((row, col), data)):
            ((r, c), sparse.csr_matrix((data, (row, col)),
                shape=(((MAX_UID - r - 1) / blk_w_num) + 1,
                    ((MAX_MID - c - 1) / blk_h_num) + 1))))

    iter_count = 0

    for i in xrange(num_iterations):
        for j in xrange(num_workers):
            target_V = V_mat.filter(lambda ((x1, x2), _): x1 == ((x2 + j) % num_workers))

            #if target_V.count() < num_workers:
            #    non_target_V = V_mat.filter(lambda ((x1, x2), _): x1 != ((x2 + j) % num_workers))
            #    non_target_V = non_target_V.map(lambda ((x1, x2), _): ((x1, x2), None))
            #    target_V = target_V.union(non_target_V)

            target_W = W_zip.map(lambda (x, _): ((x, (x - j) % num_workers), _))
            target_H = H_zip.map(lambda (x, _): (((x + j) % num_workers, x), _))
            target_W_H = target_W.join(target_H)
            V_W_H = target_V.join(target_W_H)
            res = V_W_H.map(lambda ((w_index, h_index), (V, (W, H))): ((w_index, h_index),
                    dsgd(V, W, H, w_index, h_index, beta_value, lambda_value,
                        blk_w_size, blk_h_size, iter_count))).collect()

            #print "target_V: " + str(target_V.count())
            #print "target_W: " + str(target_W.count())
            #print "target_H: " + str(target_H.count())
            #print "V_W_H: " + str(V_W_H.count())
            #print "res: " + str(res.count())

            #print res

            W_newzip = []
            H_newzip = []

            for ((w_index, h_index), (W_new, H_new, iter_count_new)) in res:
                W_newzip.append((w_index, W_new))
                H_newzip.append((h_index, H_new))
                iter_count += iter_count_new

                print w_index, W_new

            #W_zip = res.map(lambda ((w_index, h_index), (W_new, H_new, iter_count)): (w_index, W_new))
            #H_zip = res.map(lambda ((w_index, h_index), (W_new, H_new, iter_count)): (h_index, H_new))
            #iter_count += res.map(lambda (_, (W_new, H_new, iter_count)): iter_count).reduce(add)

            W_zip = sc.parallelize(W_newzip)
            H_zip = sc.parallelize(H_newzip)

            print "iteration " + str(j) + " count: " + str(iter_count)

            #print W_zip.collect()
            #print H_zip.collect()
    print W_zip.collect()
    print H_zip.collect()

def dsgd(V, W, H, w_index, h_index, beta_value, lambda_value,
        blk_w_size, blk_h_size, iter_count):
    #if not V: return W, H, iter_count

    L = nzsl(V, W, H, lambda_value)
    L_prev = sys.float_info.max
    V_loc = V.tocoo()
    sgd_count = 0

    # TODO: Introduce randomization here
    for uid, mid, rating in itertools.izip(V_loc.row, V_loc.col, V_loc.data):
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

        #if w_index == 0 and h_index == 0: print W_old_row, W[uid, :], L_prev, L
        if L >= L_prev:
            W[uid, :] = W_old_row
            H[:, mid] = H_old_col
            return W, H, sgd_count

        sgd_count += 1

        #if w_index == 0 and h_index == 0: print W_old_row, W[uid, :]

    #if w_index == 0 and h_index == 0: print "final W: ", W[uid, :]
    return W, H, sgd_count

def nzsl(V, W, H, lambda_value):
    res = 0.0
    V_loc = V.tocoo()

    for uid, mid, rating in itertools.izip(V_loc.row, V_loc.col, V_loc.data):
        res += math.pow(rating - np.dot(W[uid, :], H[:, mid]), 2)

    res += lambda_value * (sum(np.add.reduce(W * W)) + sum(np.add.reduce(H * H)))

    return res

def map_file(t):
    arr = t[1].split("\n", 1)
    mid = int(re.findall('\d+', arr[0])[0])
    tmp = [x.split(",") for x in arr[1].split("\n")]
    return [((int(elem[0]), mid), int(elem[1])) for elem in tmp if len(elem) == 3]

if __name__ == "__main__":
    main()
