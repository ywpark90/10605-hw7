# 10605-hw7
10605 HW7 (Author: Young Woong Park @youngwop)

# How to run the program
spark-submit dsgd_mf.py num_factors num_workers num_iterations beta_value lambda_value inputPath outputW_filepath outputH_filepath

ex) spark-submit dsgd_mf.py 100 10 50 0.8 1.0 test.csv w.csv h.csv

inputPath can be either file or directory
