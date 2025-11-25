"Importing modules"
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import time

"""
DONE: Read data
TODO: Make it possible to accept random_seed from the command line
TODO: Convert data to user-item matrix (movies/ratings per user)
TODO: Write function for Jaccard similarity
TODO: Write function for shingling (make a matrix with a column for each user and each row showing whether they watched the movie or not)
DONE: Write function for minhashing
TODO: Optimise function for minhashing
TODO: Write LSH algorithm function
TODO: Write a README file with instructions on how to run the file for the grader
TODO: Write the output of the LSH algorithm as user1,user2 (with user1<user2)
TODO: store signature matrix to prevent having to remake it everytime.
TODO: Test output of for different random seeds
"""

"Setting an initial value for the seed, for testing"
seed = 42
seed_max = 2025 #temporary

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

    return S_i, unique_users, unique_movies
    
# S_i, user_id, movie_id = load_data_to_sparse_matrix()

def minhash_sig(S_i, n_permutations, seed):
    "Signature matrix with shape (n_permutations, n_users)"
    n_rows, n_cols = S_i.shape
    sign_matrix = np.zeros((n_permutations, n_cols), dtype=int)

    for i in range(n_permutations):                         # for every permutation
        np.random.seed(int(i * seed))                       # allows the use of a random seed and avoids repeats in permutation of indices
        
        "Permute the rows of the sparse matrix"
        perm        = np.random.permutation(n_rows) 
        perm_sparse = S_i[perm, :]

        for j in range(n_cols):                             # for every user (column)
            start             = perm_sparse.indptr[j]       # start index of column j
            end               = perm_sparse.indptr[j + 1]   # end index of column j
            col_rows          = perm_sparse.indices[start:end]
            "First row (minimum index) where column (user) j is '1' after permutation"
            sign_matrix[i, j] = col_rows.min()

    return sign_matrix

# print("The produced signature matrix is \n",minhash_sig(S_i,5, 42))

def timing(func):
    "This function can be used as a decorator to time functions"
    def wrapper_function(*args,**kwargs): 
        start_time = time.time()
        func(*args,**kwargs)
        end_time   = time.time()

        print("This function {} took {:.2g} seconds to run".format(func.__name__,end_time - start_time))
    return wrapper_function

"""
Minhashing
"""
"Using the 'characteristic matrix' from slide 15 of lecture 4 as example"
char_matrix_test = np.array([[1,0,1,0],[1,0,0,1],[0,1,0,1],[0,1,0,1],[0,1,0,1],[1,0,1,0],[1,0,1,0]]).reshape(7,4)

def permutation_test(n_permutations,length=7,seed=seed,index=0):
    """
    This function can be used to visually check whether the indices of the matrix are actually permuted uniformly or not

    Input
    n_permutations  : the number of permutations done. Should be an integer
    seed            : the initial value of the random seed
    index           : the position along which a histogram will be made of the permuted indices
    
    Output
    A histogram

    """
    "Creating an array to keep track of how the indices are permuted"
    test_array = np.array([])

    "Assign the initial seed value"
    permutation_seed = seed

    "Writing the new permutation to an array"
    for _ in range(n_permutations):
        "Create a new permutation of the matrix"
        permutation = np.random.RandomState(seed=permutation_seed).permutation(length) #this function allows the use of a random seed and avoids repeats in permutation of indices

        "Write the permutation of the indices to an array"
        test_array  = np.append(test_array,permutation)

        "Update the seed with a new one"
        permutation_seed = np.random.randint(0,seed_max)

    "Reshape the array of the permutation indices"
    test_array = test_array.reshape(n_permutations,length)

    "Make a plot of the distribution of indices for the given index"
    plt.figure()
    plt.hist(test_array[:,index])
    plt.xlabel("Index")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of indices by permuting {n_permutations} times")
    plt.show()

@timing
def minhash_slow(char_matrix,n_permutations,seed=seed):
    """
    This function is a simple implementation of the minhash algorithm.
    
    Input
    char_matrix     : the characteristic matrix containing a column for each user. Each row represents whether a user watched that movie or not
    n_permutations  : the number of permutations done. Determines the length of the signature matrix. Should be an integer
    seed            : the value of the random seed used for the initial permutation
    
    Output
    sign_matrix     : the signature matrix with shape (n_permutations,users)
    
    """
    "The given seed will be used as an initial value. The seed will then be changed to actually create new permutations"
    permutation_seed = seed

    "Finding the number of columns of the characteristic matrix"
    users = char_matrix.shape[1]

    "Making the signature matrix"
    sign_matrix = np.zeros(int(n_permutations * users)).reshape(n_permutations,users)
    print("The signature matrix looks like this",sign_matrix)

    "Perform the different permutations and find the first nonzero indices for each user"
    for i in range(n_permutations):
        print(30*"- ")
        "Make a copy of the characteristic matrix"
        char_matrix_copy = char_matrix.copy()

        "Generate a random permutation of the indices of the rows of the characteristic matrix"
        permutation = np.random.RandomState(seed=permutation_seed).permutation(char_matrix.shape[0]) #this function allows the use of a random seed and avoids repeats in permutation of indices

        "Update the seed used for the permutation with a new one"

        permutation_seed = np.random.randint(0,seed_max)

        "Reorder the matrix, based on the permutation"
        char_matrix_copy = char_matrix_copy[permutation]

        print("The permuted matrix looks like this \n",char_matrix_copy)

        "For every user, find the first index where the column has a nonzero value"
        for j in range(users):
            print(f"Nonzero indices for user {j+1} are {np.nonzero(char_matrix_copy[:,j])[0][0]}")
            nonzero_index = np.nonzero(char_matrix_copy[:,j])[0][0]

            "And update the signature matrix with this first nonzero index"
            sign_matrix[i,j] = nonzero_index

    print("Resulting signature matrix is",sign_matrix,"with shape",sign_matrix.shape)

    return sign_matrix    


"Instead use hash functions to make the permutations? This has been used in assignment 2 task 2"

def minhash_fast(char_matrix,n_permutations,seed=seed):
    """
    TODO: write information on function
    TODO: make it possible to check multiple columns at the same time, instead of looping over each column
    """

    return None

"Runnign the permutation test to analyse its output"
# permutation_test(1000)

"Running the minhash function to analyse its output"

minhash_slow(char_matrix_test,6)

"""
Locality Sensitive Hashing
"""
def LSH():
      """ Substeps:
      - Split data of columns into bands (size of band estimated by b=n/5, or for a range of b's)
      - Hash bands into hash table
      - Sort hash table from empty to full buckets
      - Check which bands are hashed to the same bucket: this is a candidate pair
      - (For a candidate pair of bands, it is likely that the corresponding columns are similar)
      - Keep track of candidate pairs
      - If in the next, fuller, bucket there is a candidate pair that we already have: throw out
      - Finish calculating all buckets in the hash table
      - Finish iterating over all hash tables of all bands
      - Calculate the similarity of columns of candidate band pairs
      - End product for plot (slide 48 L4): Similarity value per candidate pair
      - End product for assignment: candidate pairs with similarity above threshold
      """


""" Testing """
testing = True
if testing:
    "Testing out the representation of a sparse matrix"
    test_matrix        = np.zeros(10000).reshape(100,100)
    test_matrix[51,61] = 1
    test_matrix[9,51]  = 3

    "Make the spare matrix"
    test = sparse.csc_matrix(test_matrix)
    print("Sparse matrix representation: \n",test)

    "Apply the load_data_to_sparse_matrix function to a simple dataset"
    def load_data_to_sparse_matrix_test():
        "Loading the data and assigning to variables"
        user_id = np.array([1,1,1,3,4,5,6])
        movie_id = np.array([20,6,5,7,6,5,6])
        
        print(f"Provided user ids are {user_id}")
        print(f"Provided movie ids are {movie_id}")

        "Store unique users and movies, and original indices"
        unique_users, user_index   = np.unique(user_id, return_inverse=True)
        unique_movies, movie_index = np.unique(movie_id, return_inverse=True)
        n_users                    = unique_users.size
        n_movies                   = unique_movies.size
        print(f"n_users = {n_users}, unique_users = {unique_users}")
        print(f"n_movies = {n_movies}, unique_movies = {unique_movies}")

        '''Binary/boolean user-item matrix: CSC matrix with shape (n_movies, n_users). 
        Entry is True if user rated movie, else not explicitly stored'''
        S_i = sparse.csc_matrix(
            (np.ones(user_index.size, dtype=bool), (movie_index, user_index)),
            shape=(n_movies, n_users),
            dtype=bool
        )

        return S_i, unique_users, unique_movies
    print("The produced sparse signature matrix is \n",load_data_to_sparse_matrix_test()[0])


    "Using an example from the lectures"    
    sig_matrix = np.array([[2,1,2,1],[2,1,4,1],[1,2,1,2]]).reshape(3,4)
    

    def user_similarity(sig_matrix,u1,u2):
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
        
        similarity = np.count_nonzero(sig_matrix[:,u1]==sig_matrix[:,u2])/len(sig_matrix[:,u1])

        return similarity
        
    # print(user_similarity(sig_matrix,0,2))

    def hash_value_calculator(vector):
        """
        Hashes a vector to a hash-value. The same hash function will be used for all bands.
        This is done, because the similarity of a band should not be dependent on the chosen
        hash function.

        Input
        vector      : 1-D array. Contains the values of one band from the signature matrix

        Output
        hash_value  : Integer. The determined hash value based on the given vector

        TODO        : Make sure this function doesn't cause too many collisions, but also not too few
        """
        hash_value = np.sum(vector) % 1024

        # print(f"Sum of the vector is {np.sum(vector)} and the hash value is {hash_value}") #for testing

        return hash_value
    
    print(sig_matrix[0:2])

    sig_matrix2 = np.array([
        [1,0,0,1,3,4],
        [2,5,2,1,3,2],
        [3,6,5,4,1,2],
        [4,1,5,4,2,1],
        [5,8,6,0,1,2],
        [6,4,8,1,6,2],
        [7,3,2,1,4,2],
        [8,1,5,2,4,5]
    ])

    def sig_to_band(signature_matrix,b):
        "Divides the signature matrix up into b smaller matrices with an equal number of rows"
        signatures, users = signature_matrix.shape

        "Make a list to store the bands of the signature matrix in"
        bands = []

        "Determine the number of rows that go in one band"
        r = signatures / b

        # print("Value for r is",r)
        "Select the band from the signature matrix and write it to a list"
        for i in range(b):
            band = signature_matrix[int(i*r):int((i+1)*r)]
            # print("Selected band is",band,"which has shape",band.shape)
            bands.append(band)
            
        return bands

        
    # sig_to_band(sig_matrix2,4)

    def hash_table_maker(signature_matrix,b):
        """
        Makes a hash table using the hash_value_calculator function.

        Input
        signature_matrix    : 2D array. The signature matrix produced by minhashing.
        b                   : Integer. Number of bands.
        
        Output 
        hash_table          : 2D array with shape (b,users). Each row is the hash table for a band. 
        """

        "Retrieve the number of users from the signature matrix"
        users = signature_matrix.shape[1]

        "Create the hash table"
        hash_table = np.zeros(int(b*users)).reshape(b,users)

        "Select the bands from the signature matrix"
        bands = sig_to_band(signature_matrix,b)

        "Looping over each band"
        for i in range(b):
            "Select the band of interest"
            band = bands[i]
            # print("The band of interest is",band)
            "Looping over each user"
            for j in range(users):
                "Calculate the hash-value and enter it into the hash_table"
                # print("The row used for the calculation is",band[:,j])
                hash_table[i,j] = hash_value_calculator(band[:,j])
        
        return hash_table
    
    print("The produced hash-table is\n",hash_table_maker(sig_matrix2,4))

    def find_candidate_pairs(hash_table):
        """
        Finds a candidate pair of users (using non-zero indices). If two users have similar hash-values
        they are considered candidate pairs
        """

        "Make a list to keep track of the different cadidate users"
        candidate_pairs = []

        "For each row in the hash table"
        for i in range(hash_table.shape[0]):
            "Select a row from the hash table"
            row = hash_table[i]
            print("Row of interest is",row)

            "Find the unique values in that row"
            uniques = np.unique(row)

            "Only continue if there is a non-unique value"
            if len(uniques) != len(row):
                "For each unique value, count its occurence in the row of the hash table"
                for unique_value in uniques:
                    "Determine the occurence"
                    occurence = len(np.nonzero(row == unique_value)[0])
                    
                    "If this is larger than one, store the users"
                    if occurence > 1:
                        "Find the indices of the non-unique values"
                        indices = np.argwhere(row == unique_value).reshape(1,2)[0]
                
                        "Convert the indices into users, then into a tuple and store in the list"
                        candidate_pairs.append(tuple(indices + 1))
                        

        print(candidate_pairs)
        return candidate_pairs 
    
    find_candidate_pairs(hash_table_maker(sig_matrix2,4))
