# -*- coding: utf-8 -*-
'''
@Time    : 2020/8/12 10:01
@Author  : daluzi
@File    : ItemCF_pyspark.py
'''

import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import random
import csv
import pdb
from pyspark import SparkConf, SparkContext

def loadMovieNames():
    '''
    Parse through the u.item file and extracts movie and user information
    '''
    movieNames= {}
    with open("ml-100k/u.item") as f:
        for line in f:
            fields= line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames
def fetchData(line):
    '''
    Fetch user , movie name and rating in a group
    '''
    line = line.split()
    return line[0],(line[1],float(line[2]))
def findMoviePairs(user_id, movie_with_rating):
    '''
    finds pair of movies and groups the ratings
    '''

    for movie1,movie2 in combinations(movie_with_rating,2):
      return (movie1[0],movie2[0]),(movie1[1],movie2[1])
def findUserPairs(item_id,users_with_rating):
    '''
    For each movie, find all user with same movie
    '''
    for user1,user2 in combinations(users_with_rating,2):
        return (user1[0],user2[0]),(user1[1],user2[1])
def keyOfFirstItem(movie_pair, movie_sim_data):
    '''
    For each user-user combiantion, make the first user's id as a key
    '''
    (movie1_id,movie2_id) = movie_pair
    return movie1_id,(movie2_id,movie_sim_data)
def cosineSim(movie_pair, rating_pairs):
    '''
    For each user-user pair, return the similarity score and co-rater score.
    '''
    sum_x, sum_xy, sum_y, x = (0.0, 0.0, 0.0, 0)

    for rating_pair in rating_pairs:
        sum_x += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_y += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])

        x += 1

    cosine_sim = cosine(sum_xy,np.sqrt(sum_x),np.sqrt(sum_y))
    return movie_pair, (cosine_sim,x)
def cosine(dot_product,rating1_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors
    '''
    num = dot_product
    den = rating1_norm_squared * rating2_norm_squared

    return (num / (float(den))) if den else 0.0
def nearNeighbors(movie_id, movie_and_sims, n):
    '''
    Sort the movie predictions list by similarity and select the top N related users
    '''
    movie_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    return movie_id, movie_and_sims[:n]
def topMovieRecommendations(user_id, movie_with_rating, movie_sims, n):
    '''
    Calculate the top N movie recommendations for each user using the
    weighted sum approach
    '''

    # initialize dicts to store the score of each individual movie and movie can exist with more than one movie
    t = defaultdict(int)
    sim_sum = defaultdict(int)

    for (movie,rating) in movie_with_rating:

        # lookup the nearest neighbors for this movie
        near_neigh = movie_sims.get(movie,None)

        if near_neigh:
            for (neigh,(sim,count)) in near_neigh:
                if neigh != movie:

                    # update totals and sim_sum
                    t[neigh] += sim * rating
                    sim_sum[neigh] += sim

    # create the normalized list of scored movies
    scored_movies = [(total/sim_sum[movie],movie) for movie,total in t.items()]

    # sort the scored movies in ascending order
    scored_movies.sort(reverse=True)


    # ranked_items = [x[1] for x in scored_items]

    return user_id,scored_movies[:n]


conf = SparkConf().setMaster("local").setAppName("Item Based Collaborative Filtering")
sc = SparkContext(conf=conf)
lines = sc.textFile("file:///SparkCourse/ml-100k/u1.base")
user_movie_pairs = lines.map(fetchData).groupByKey().cache()

paired_movies = user_movie_pairs.filter(
lambda p: len(p[1]) > 1).map(
lambda p: findMoviePairs(p[0], p[1])).groupByKey()

movie_sim = paired_movies.map(
        lambda p: cosineSim(p[0], p[1]))

movie_sim=movie_sim.map(
        lambda p: keyOfFirstItem(p[0], p[1])).groupByKey()

movie_sim=movie_sim.map(lambda x : (x[0], list(x[1]))).map(
        lambda p: nearNeighbors(p[0], p[1], 3)).collect()

movie_sim_dict = {}
for (movie,data) in movie_sim:
    movie_sim_dict[movie] = data
i = sc.broadcast(movie_sim_dict)
user_movie_recs = user_movie_pairs.map(
lambda p: topMovieRecommendations(p[0], p[1], i.value, 3)).collect()
nameDict = loadMovieNames()

result= user_movie_recs
movieList = list()
'''
print top recommended movies for each user
'''
for r in result:
    (user, movie_pair) = r
    for (rating, movieid) in movie_pair:
        movieList.append(nameDict[int(movieid)])
    print("For user ",user, "movie recommendations are ", movieList)
    del movieList[:]
    print("end of tuple")
        #movieList.append(nameDict[int(p)])
    '''
    print "For user id ",user, "movie recommendations are ", movieList
    del movieList[:]
    print "end of tuple"
    '''