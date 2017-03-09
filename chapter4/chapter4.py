from __future__ import division

height_weight_age = [70, 170, 40]
print height_weight_age

grades = [95, 80, 75, 62]
print grades

def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]

v = [1, 2]
w = [3, 4]
print vector_add(v, w)

def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]

print vector_subtract(v, w)

def vector_sum(vectors):
    result = vectors[0]
    for vector in vectors[1:]:
        result = vector_add(result, vector)
    return result

print vector_sum([v, w])

def vector_sum(vectors):
    return reduce(vector_add, vectors)

print vector_sum([v, w])

from functools import partial

vector_sum = partial(reduce, vector_add)
print vector_sum([v, w])

def scala_multiply(c, v):
    return [c * v_i for v_i in v]

print scala_multiply(2, v)

def vector_mean(vectors):
    n = len(vectors)
    return scala_multiply(1/n, vector_sum(vectors))

print vector_mean([v, w])

def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

print dot(v, w)

def sum_of_squares(v):
    return dot(v, v)

print sum_of_squares(v)

import math

def magnitude(v):
    return math.sqrt(sum_of_squares(v))

print magnitude(v)

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

print squared_distance(v, w)

def distance(v, w):
    return math.sqrt(squared_distance(v, w))

print distance(v, w)

def distance(v, w):
    return magnitude(vector_subtract(v, w))

print distance(v, w)

A = [[1, 2, 3],
     [4, 5, 6]]
print A

B = [[1, 2],
     [3, 4],
     [5, 6]]
print B

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

print shape(A)
print shape(B)

def get_row(A, i):
    return A[i]

def get_column(A, j):
    return [A_i[j] for A_i in A]

print get_row(A, 0)
print get_column(A, 0)

def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j)
             for j in range(num_cols)]
            for i in range(num_rows)]

def is_diagonal(i, j):
    return 1 if i == j else 0

identity_matrix = make_matrix(5, 5, is_diagonal)
print identity_matrix

friendships = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # user 0
               [1, 0, 1, 1, 0, 0, 0, 0, 0, 0], # user 1
               [1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # user 2
               [0, 1, 1, 0, 1, 0, 0, 0, 0, 0], # user 3
               [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # user 4
               [0, 0, 0, 0, 1, 0, 1, 1, 0, 0], # user 5
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 6
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 7
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 1], # user 8
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] # user 9

print friendships[0][2] == 1
print friendships[0][8] == 1

friends_of_five = [i for i, is_friend in enumerate(friendships[5]) if is_friend]
print friends_of_five
