# -*- coding: UTF-8 -*-
import numpy as np
import sys
from pyspark import SparkContext

output_file = sys.argv[1]
input_file = sys.argv[2]

sc = SparkContext(appName="test")
rdd = sc.textFile(input_file)
result = rdd.collect()
result_final = []

for i, item in enumerate(result):
    print(i, item)
    if float(item) < 0.5:
        result_final.append(i+1)

print('machine nums:', len(result_final))

result_rdd = sc.parallelize(result_final)
result_rdd.saveAsTextFile(output_file)