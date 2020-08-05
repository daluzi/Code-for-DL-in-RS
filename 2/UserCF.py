# -*- coding: utf-8 -*-
'''
@Time    : 2020/8/5 15:56
@Author  : daluzi
@File    : UserCF.py
'''

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

moviesPath = "../dataset/ml-latest-small/movies.csv"
ratingsPath = "../dataset/ml-latest-small/ratings.csv"
moviesDF = pd.read_csv(moviesPath, index_col=None)
ratingsDF = pd.read_csv(ratingsPath, index_col=None)

trainRatingsDF, testRatingsDF = train_test_split(ratingsDF, test_size=0.2)
print("total_movie_count:" + str(len(set(ratingsDF['movieId'].values.tolist()))))
print("total_user_count:" + str(len(set(ratingsDF['userId'].values.tolist()))))
print("train_movie_count:" + str(len(set(trainRatingsDF['movieId'].values.tolist()))))
print("test_movie_count:" + str(len(set(testRatingsDF['movieId'].values.tolist()))))
print("train_user_count:" + str(len(set(trainRatingsDF['userId'].values.tolist()))))
print("test_user_count:" + str(len(set(testRatingsDF['userId'].values.tolist()))))


trainRatingsPivotDF = pd.pivot_table(trainRatingsDF[['userId', 'movieId', 'rating']], columns=['movieId'],
                                 index=['userId'], values='rating', fill_value=0)

# enumerate返回穷举序列号与值
# 8975部电影
moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))
# 610个用户
usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))
# 矩阵变成list 每一行变成list的一个值 长度为610 每个值大小为8975
ratingValues = trainRatingsPivotDF.values.tolist()


def calCosineSimilarity(list1, list2):
    res = 0
    denominator1 = 0
    denominator2 = 0
    for (val1, val2) in zip(list1, list2):
        res += (val1 * val2)
        denominator1 += val1 ** 2
        denominator2 += val2 ** 2
    return res / (math.sqrt(denominator1 * denominator2))

# 根据用户对电影的评分，来判断每个用户间相似度
userSimMatrix = np.zeros((len(ratingValues), len(ratingValues)), dtype=np.float32)
for i in range(len(ratingValues) - 1):
    for j in range(i + 1, len(ratingValues)):
        userSimMatrix[i, j] = calCosineSimilarity(ratingValues[i], ratingValues[j])
        userSimMatrix[j, i] = userSimMatrix[i, j]

# 找到与每个用户最相近的前K个用户
userMostSimDict = dict()
for i in range(len(ratingValues)):
    userMostSimDict[i] = sorted(enumerate(list(userSimMatrix[i])), key=lambda x: x[1], reverse=True)[:10]

# 用这K个用户的喜好中目标用户没有看过的电影进行推荐
userRecommendValues = np.zeros((len(ratingValues), len(ratingValues[0])), dtype=np.float32)  # 610*8975

for i in range(len(ratingValues)):
    for j in range(len(ratingValues[i])):
        if ratingValues[i][j] == 0:
            val = 0
            for (user, sim) in userMostSimDict[i]:
                val += (ratingValues[user][j] * sim)
            userRecommendValues[i, j] = val

userRecommendDict = dict()
for i in range(len(ratingValues)):
    userRecommendDict[i] = sorted(enumerate(list(userRecommendValues[i])), key=lambda x: x[1], reverse=True)[:10]

# 将一开始的索引转换为原来用户id与电影id
userRecommendList = []
for key, value in userRecommendDict.items():
    user = usersMap[key]
    for (movieId, val) in value:
        userRecommendList.append([user, moviesMap[movieId]])

# 将推荐结果的电影id转换成对应的电影名
recommendDF = pd.DataFrame(userRecommendList, columns=['userId', 'movieId'])
recommendDF = pd.merge(recommendDF, moviesDF[['movieId', 'title']], on='movieId', how='inner')
print(recommendDF.tail(10))