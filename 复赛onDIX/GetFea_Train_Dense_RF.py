# -*- coding: UTF-8 -*-

# Log:
# 错误 UnicodeDecodeError: 'ascii' codec can't decode byte 0xe5 in position 677: ordinal not in range(128)
# 原因：对长度只有1的数组逐次相减了 或者 对单个数进行了计算均值、方差、最小值之类的操作



# from __future__ import print_function
import sys
from pyspark import SparkContext



import sys
reload(sys)
sys.setdefaultencoding('UTF-8')

# DIX用
output_file = sys.argv[1]
input_file = sys.argv[2]


def fea_extra_one(a_sample_str):
    import numpy as np

    np.set_printoptions(precision=8)

    # 1.解析原始数据
    # a_sample_str.decode("utf-8")
    try:
        a_sample_str = a_sample_str.encode("utf-8")  # omit in 3.x!
    except UnicodeEncodeError:
        pass

    part = a_sample_str.split(" ")
    id = part[0]
    trace = part[1]
    aim = part[2]
    aim = aim.split(',')
    aim = (float(aim[0]), float(aim[1]))
    label = int(part[3])

    # 提取特征部分省略

    # 随机森林训练时，训练集需要加上label，放在最后一列
    fea_list.append(label)

    # 输出成Dense格式
    fea_str_list = [str(item) for item in fea_list]
    fea_list = ' '.join(fea_str_list)

    return fea_list


sc = SparkContext(appName="test")
rdd = sc.textFile(input_file)

result = rdd.map(fea_extra_one)
print(result.take(2))
result.saveAsTextFile(output_file)

print("Done!")