# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:11:23 2023

@author: mud, Chel, Nick
"""

from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from collections import defaultdict
import time
import statistics


def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if int(v) > int(u):
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if int(w) > int(v) and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count


def MR_ApproxTCwithNodeColors(edges, C):
    # define hash equation variables in this scope to have the same vars for every calculations.
    p = 8191
    a = rand.randint(1, p-1)
    b = rand.randint(0, p-1)

    def hash_color(pair):
        def run_equation(vertex):
            return ((a*int(vertex) + b) % p) % int(C)

        u, v = pair

        hcu = run_equation(u)
        hcv = run_equation(v)

        # Check if endpoints have the same hash score
        if hcu == hcv:
            return [(hcu, (u, v))]
        else:
            return []

    t_count = edges.flatMap(lambda x: hash_color(x)) \
        .groupByKey() \
        .flatMap(lambda x: [(0, CountTriangles(x[1]))]) \
        .reduceByKey(lambda x, y: x+y)
    
    t_final = C*C*t_count.collect()[0][1]
    
    return t_final


def splitPair(pair):
    u, v = map(int, pair.split(','))
    return [(u, v)]


def MR_ApproxTCwithSparkPartitions(edges, C):
    t_count = edges \
        .mapPartitions(lambda partition: [CountTriangles(list(partition))]) \
        .reduce(lambda count1, count2: count1 + count2)

    t_final = C*C*t_count

    return t_final


def main():

    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 4, "Usage: python hw1.py <C> <R> <file_name>"

    # SPARK SETUP
    conf = SparkConf().setAppName('G096HW1')
    sc = SparkContext(conf=conf).getOrCreate()

    # INPUT READING
    # 1. Read number of colours
    C = sys.argv[1]
    assert C.isdigit(), "C must be an integer"
    C = int(C)

    # 2. Read number of runs
    R = sys.argv[2]
    assert R.isdigit(), "R must be an integer"
    R = int(R)

    # 3. Read input graph
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"

    rawData = sc.textFile(data_path)
    
    # Transform the RDD of strings into an RDD of edges, partition and cache them.
    edges = rawData.flatMap(splitPair).repartition(numPartitions=C).cache()
    

    # SETTING GLOBAL VARIABLES
    print("Dataset = ", os.path.basename(data_path))
    print("Number of Edges = ", edges.count())
    print("Number of Colors = ", C)
    print("Number of Repetitions = ", R)
    print("Approximation through node coloring")

    # # 1-TRIANGLE COUNT WITH COLORS
    t_final_estimates = []
    start_time = time.perf_counter()
    
    for i in range(R):
        t_final_estimates.append(MR_ApproxTCwithNodeColors(edges, C))

    total_time_ms = ((time.perf_counter() - start_time) * 1000)/R
    
    print("- Number of triangles (median over", R, "runs) = ",
          statistics.median(t_final_estimates))
    print("- Running time (average over", R, "runs) = ",
          int(total_time_ms), " ms")

    # 2 - SPARK PARTITIONING with reduce
    start_time = time.perf_counter()
    partitionCount = MR_ApproxTCwithSparkPartitions(edges, C)
    total_time_ms = (time.perf_counter() - start_time) * 1000
    print("Approximation through Spark partitioning")
    print('- Number of triangles = ', partitionCount)
    print("- Running time = ",
          int(total_time_ms), " ms")


    sc.stop()


if __name__ == "__main__":
    main()
