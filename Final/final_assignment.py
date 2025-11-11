"Importing modules"
import numpy as np
import matplotlib.pyplot as plt

"""
TODO: Read data
TODO: Convert data to user-item matrix (movies/ratings per user)
TODO: Write function for Jaccard similarity
TODO: Write LSH algorithm function
"""

data = np.load('user_movie_rating.npy')
user_id, movie_id, rating = data[:,0], data[:,1], data[:,2]

