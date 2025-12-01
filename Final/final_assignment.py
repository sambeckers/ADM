"Importing modules"
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import time
from tqdm import tqdm
import argparse
import signal
import sys
import sys

"""
DONE: Read data
DONE: Make it possible to accept random_seed from the command line
DONE: Convert data to user-item matrix (movies/ratings per user)
DONE: Write function for Jaccard similarity
DONE: Write function for shingling (make a matrix with a column for each user and each row showing whether they watched the movie or not)
DONE: Write function for minhashing
DONE: Optimise function for minhashing
DONE: Write LSH algorithm function
DONE: Write a README file with instructions on how to run the file for the grader
DONE: Write the output of the LSH algorithm as user1,user2 (with user1<user2)
DONE: store signature matrix to prevent having to remake it everytime.
DONE: Test output of for different random seeds
DONE: Append results in result.txt file, and close after (see hint 6)
"""

"Timeout handler"
class TimeoutError(Exception):
    pass

similarities_global = {}
output_file_global = 'results.txt'

def timeout_handler(signum, frame):
    print('\nExecution exceeded 30 minutes - saving partial results...')
    if similarities_global:
        with open(output_file_global, 'w') as f:
            for (u1, u2), sim in sorted(similarities_global.items()):
                f.write('{},{},{:.4f}\n'.format(u1, u2, sim))
            f.flush()
        print('Saved {} pairs to {}'.format(len(similarities_global), output_file_global))
    else:
        print('No pairs found yet')
    sys.exit(0)

"Setting an initial value for the seed, for testing"
seed = 42

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
    sign_matrix    = np.zeros((n_permutations, n_cols), dtype=int)

    print('Creating minhash signature matrix with {} permutations...'.format(n_permutations))
    for i in tqdm(range(n_permutations), total=n_permutations): # for every permutation
        np.random.seed(int(i * seed)) # allows the use of a random seed and avoids repeats in permutation of indices
        
        "Permute the rows of the sparse matrix"
        perm        = np.random.permutation(n_rows) 
        perm_sparse = S_i[perm, :]

        for j in range(n_cols): # for every user (column)
            start             = perm_sparse.indptr[j]       # start index of column j
            end               = perm_sparse.indptr[j + 1]   # end index of column j
            col_rows          = perm_sparse.indices[start:end]
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
            # Use smaller modulo to prevent overflow with large r values
            if r > 5:
                # For large r, use tuple hashing instead
                hashes = np.array([hash(tuple(band[:, j])) for j in range(band.shape[1])])
            else:
                hashes = np.ravel_multi_index(multi_index=band, dims=band.max(axis=1)+1)
        except (ValueError, OverflowError) as e:
            # Fallback to tuple hashing if ravel_multi_index fails
            hashes = np.array([hash(tuple(band[:, j])) for j in range(band.shape[1])])
       
        "Sort the hash values"
        sort_user_idx = np.argsort(hashes) # These are the USER INDICES sorted by hash value!
        hashes        = hashes[sort_user_idx]

        "Find where hash values change (bucket boundaries), "
        "nonzero finds indices where condition (adjacent value is larger than current) is True"
        boundaries    = np.nonzero(hashes[1:] > hashes[:-1])[0] + 1
        
        "Split into buckets - each bucket contains user indices with same hash"
        buckets_arr   = np.split(sort_user_idx, boundaries)

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
    found_pairs    = set()
    verified_pairs = set()

    "Sort buckets by size (smallest first)"
    buckets_sorted = sorted(buckets, key=lambda x: len(x))
    print('Largest bucket has size: {}'.format(len(buckets_sorted[-1])))

    print('Finding candidate user pairs from LSH buckets (small to large)...')
    for bucket in tqdm(buckets_sorted, total=len(buckets_sorted)):

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
    similarities = {}

    print('Verifying candidate user pairs...')
    for u1, u2 in tqdm(found_pairs, total=len(found_pairs)):
        sim = jaccard_similarity(sparse_mat, u1, u2)
        if sim > threshold:
            verified_pairs.add((u1, u2))
            similarities[(u1, u2)] = sim
            similarities_global[(u1, u2)] = sim

    print('Number of similar user pairs found: {}'.format(len(verified_pairs)))

    return verified_pairs, similarities


def main(seed, b, r, n_permutations, threshold, output_file):
    global similarities_global, output_file_global
    similarities_global = {}
    output_file_global = output_file
    
    if n_permutations < b * r:
        print('Error: n_permutations ({}) must be >= b*r ({})'.format(n_permutations, b*r))
        sys.exit(1)
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30 * 60)
    
    start_time = time.time()
    print('Running LSH pipeline with seed={}, b={}, r={}, n_permutations={}, threshold={}\n'.format(
        seed, b, r, n_permutations, threshold
    ))

    S_i               = load_data_to_sparse_matrix()
    sig_matrix        = minhash_sig(S_i=S_i, n_permutations=n_permutations, seed=seed)
    buckets           = lsh(sig_matrix=sig_matrix, b=b, r=r)
    verified_pairs, similarities    = find_similar_user_pairs(
        buckets       = buckets,
        sig_matrix    = sig_matrix,
        sparse_matrix = S_i,  
        threshold     = threshold
    )
    
    with open(output_file, 'w') as f:
        for (u1, u2), sim in sorted(similarities.items()):
            f.write('{},{},{:.4f}\n'.format(u1, u2, sim))
    
    elapsed_time = (time.time() - start_time) / 60
    print('Total execution time: {:.2f} minutes'.format(elapsed_time))
    
    signal.alarm(0)
    return verified_pairs, similarities

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSH for finding similar users')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--b', type=int, default=20, help='Number of bands')
    parser.add_argument('--r', type=int, default=5, help='Rows per band')
    parser.add_argument('--n', type=int, default=None, help='Number of permutations (default: b*r)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold')
    parser.add_argument('--output', type=str, default='results.txt', help='Output file')
    
    args = parser.parse_args()
    
    # Automatically set n = b * r if not specified
    if args.n is None:
        args.n = args.b * args.r
        print('Setting n_permutations = b * r = {}'.format(args.n))
    
    results, sims = main(seed=args.seed, b=args.b, r=args.r, n_permutations=args.n,
                         threshold=args.threshold, output_file=args.output)
