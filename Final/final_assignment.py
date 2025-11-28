"Importing modules"
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import time
from tqdm import tqdm
import sys

"""
DONE: Read data
TODO: Make it possible to accept random_seed from the command line
DONE: Convert data to user-item matrix (movies/ratings per user)
DONE: Write function for Jaccard similarity
TODO: Write function for shingling (make a matrix with a column for each user and each row showing whether they watched the movie or not)
DONE: Write function for minhashing
TODO: Optimise function for minhashing
TODO: Write LSH algorithm function
TODO: Write a README file with instructions on how to run the file for the grader
TODO: Write the output of the LSH algorithm as user1,user2 (with user1<user2)
TODO: store signature matrix to prevent having to remake it everytime.
TODO: Test output of for different random seeds
TODO: Append results in result.txt file, and close after (see hint 6)
"""

"Setting an initial value for the seed, for testing"
seed = 42
seed_max = 2025 #temporary

"""
Data loading and processing
"""
def load_data_to_sparse_matrix():
    "Loading the data and assigning to variables"
    data                = np.load('user_movie_rating.npy')
    user_id, movie_id   =  data[:,0], data[:,1] # Do not load ratings as they are irrelevant
    
    "Store unique users and movies, and original indices"
    unique_users, user_index   = np.unique(user_id, return_inverse=True)
    unique_movies, movie_index = np.unique(movie_id, return_inverse=True)
    n_users                    = unique_users.size
    n_movies                   = unique_movies.size

    '''Binary/boolean user-item matrix: CSC matrix with shape (n_movies, n_users). 
    Entry is True if user rated movie, else not explicitly stored'''
    S_i = sparse.csc_matrix(
        (np.ones(user_index.size, dtype=bool), (movie_index, user_index)),
        shape=(n_movies, n_users),
        dtype=bool
    )
    print('Data loaded: {} users, {} movies\n'.format(n_users, n_movies))
    return S_i

"""
Minhashing
"""
def minhash_sig(S_i, n_permutations, seed):
    "Signature matrix with shape (n_permutations, n_users)"
    n_rows, n_cols = S_i.shape
    sign_matrix = np.zeros((n_permutations, n_cols), dtype=int)

    print('Creating minhash signature matrix with {} permutations...'.format(n_permutations))
    for i in tqdm(range(n_permutations), total=n_permutations): # for every permutation
        np.random.seed(int(i * seed)) # allows the use of a random seed and avoids repeats in permutation of indices
        
        "Permute the rows of the sparse matrix"
        perm = np.random.permutation(n_rows) 
        perm_sparse = S_i[perm, :]

        for j in range(n_cols): # for every user (column)
            start = perm_sparse.indptr[j]       # start index of column j
            end = perm_sparse.indptr[j + 1]   # end index of column j
            col_rows = perm_sparse.indices[start:end]
            "First row (minimum index) where column (user) j is '1' after permutation"
            sign_matrix[i, j] = col_rows.min()

    print('Minhash signature matrix created.\n')
    return sign_matrix

"""
Jaccard Similarity approximation
"""
def lsh_user_similarity(sig_matrix,u1,u2):
    """
    Approximates the Jaccard similarity. This is done by counting how many elements
    in the columns of the signature matrix for two users are equal to each other.
    This number is compared by the length of the column. This approximates the Jaccard
    similarity because of the many permutations of the characteristic matrix

    Input
    sig_matrix  : 2D array. The signature matrix produced by minhashing.
    u1          : 1D array. The column of user 1 in the signature matrix / band
    u2          : 1D array. The column of user 2 in the signature matrix / band

    Output
    similarity  : Float. The similarity between two users
    """
    return np.count_nonzero(sig_matrix[:,u1]==sig_matrix[:,u2])/len(sig_matrix[:,u1])

def jaccard_similarity(sparse_matrix, u1, u2):
    set1 = sparse_matrix[:, u1]
    set2 = sparse_matrix[:, u2]
    return np.sum(set1 & set2) / np.sum(set1 | set2)

"""
Local Sensitivity Hashing
"""
def lsh(sig_matrix, b, r):

    buckets = []

    print('Performing LSH with {} bands and {} rows per band...'.format(b, r))
    for i in tqdm(range(b), total=b):
        "Divide the signature matrix into b bands consisting of r rows each"
        "Take current row, and the next r rows. Then shift to the next rows for next iter"
        band = sig_matrix[i*r:(i+1)*r]

        "Hash the indices of each band, with dimension equal to the maximum value in each row"
        try:
            hashes = np.ravel_multi_index(multi_index=band, dims=band.max(axis=1)+1)
        except ValueError as e:
            print(e)
            print('b*r exceeds signature matrix dimensions. Run again with n > b*r.')
            sys.exit(1)
       
        "Sort the hash values"
        sort_user_idx = np.argsort(hashes) # These are the USER INDICES sorted by hash value!
        hashes = hashes[sort_user_idx]

        "Find where hash values change (bucket boundaries), "
        "nonzero finds indices where condition (adjacent value is larger than current) is True"
        boundaries = np.nonzero(hashes[1:] > hashes[:-1])[0] + 1
        
        "Split into buckets - each bucket contains user indices with same hash"
        buckets_arr = np.split(sort_user_idx, boundaries)

        "Only keep buckets with more than one user, cannot compare single users"
        for j in range(len(buckets_arr)):
            if len(buckets_arr[j]) > 1:
                buckets.append(buckets_arr[j])

    "Remove duplicate buckets: convert to set (no duplicates) from tuples (hashable)"
    buckets = list(set(map(tuple, buckets)))

    return buckets

"""
Find pairs of similar users from LSH buckets
"""
def find_similar_user_pairs(buckets, sig_matrix, sparse_matrix, threshold=0.5) -> None:
    "Store the already seen pairs in unordered set"
    found_pairs = set()
    verified_pairs = set()

    "Collect unique unordered sets from the bucket"

    print('Finding candidate user pairs from LSH buckets...')
    for bucket in tqdm(buckets, total=len(buckets)):

        "Go through all values in the bucket"
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                u1, u2 = bucket[i], bucket[j]

                "Avoid duplicate pairs, set u1<u2"
                if u1 > u2:
                    u1, u2 = u2, u1
                    
                "Check if pair already found"
                if (u1, u2) not in found_pairs:

                    "Similarity threshold calculation"
                    if lsh_user_similarity(sig_matrix, u1, u2) > threshold:
                        found_pairs.add((u1, u2))

    "Are candidate pairs really similar? Compare signatures and original objects"
    sparse_mat = sparse_matrix.toarray()

    print('Verifying candidate user pairs...')
    for u1, u2 in tqdm(found_pairs, total=len(found_pairs)):
        if jaccard_similarity(sparse_mat, u1, u2) > threshold:
            verified_pairs.add((u1, u2))

    print('Number of similar user pairs found: {}'.format(len(verified_pairs)))

    with open('results.txt', 'w') as f:
        for u1, u2 in sorted(verified_pairs):
            f.write('{},{}\n'.format(u1, u2))


def main(seed, n_permutations, b, r, threshold):
    start_time = time.time()
    print('Running LSH pipeline with seed={}, n_permutations={}, b={}, r={}, threshold={}\n'.format(
        seed, n_permutations, b, r, threshold
    ))

    S_i = load_data_to_sparse_matrix()
    sig_matrix = minhash_sig(S_i=S_i, n_permutations=n_permutations, seed=seed)
    buckets = lsh(sig_matrix=sig_matrix, b=b, r=r)
    verified_pairs = find_similar_user_pairs(
        buckets=buckets,
        sig_matrix=sig_matrix,
        sparse_matrix=S_i,  
        threshold=threshold
    )
    
    elapsed_time = (time.time() - start_time) / 60
    print('Total execution time: {:.2f} minutes'.format(elapsed_time))
    
    return verified_pairs

if __name__ == '__main__':
    # Run the pipeline
    results = main(seed=42, n_permutations=100, b=33, r=3, threshold=0.5)
