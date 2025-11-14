"Importing modules"
import numpy as np
import matplotlib.pyplot as plt

"""
TODO: Read data
TODO: Make it possible to accept random_seed from the command line
TODO: Convert data to user-item matrix (movies/ratings per user)
TODO: Write function for Jaccard similarity
TODO: Write LSH algorithm function
TODO: Write a README file with instructions on how to run the file for the grader
"""


"Loading the data and assigning to variables"
data = np.load('user_movie_rating.npy')
user_id, movie_id, rating = data[:,0], data[:,1], data[:,2]

"""
Minhashing
"""
"Using the 'characteristic matrix' from slide 15 of lecture 4 as example"
char_matrix_test = np.array([[1,0,1,0],[1,0,0,1],[0,1,0,1],[0,1,0,1],[0,1,0,1],[1,0,1,0],[1,0,1,0]]).reshape(7,4)

"Each row is a 'shingle' which in this case is just whether a person rated a movie or not"
length = char_matrix_test.shape[0]

"Introducing some settings for the random seeds"
seed     = 0 # a value for the random seed
seed_max = 2025 # a maximum value of the seed

def minhash_initial(char_matrix,n_permutations,seed=seed):
    test_array = np.array([])
    permutation_seed = seed
    "Finding the number of columns of the characteristic matrix"
    length = char_matrix.shape[1]

    "Making the signature matrix"
    sign_matrix = np.zeros(int(n_permutations * length)).reshape(n_permutations,length)
    print("The signature matrix looks like this",sign_matrix)

    for i in range(n_permutations): # for every permutation
        permutation = np.random.RandomState(seed=permutation_seed).permutation(length) #this function allows the use of a random seed and avoids repeats in permutation of indices

        "Writing the new permutation to an array"
        test_array  = np.append(test_array,permutation)

        "Update the seed with a new one"
        permutation_seed = np.random.randint(0,seed_max)

        "Permuting the matrix"
        "NOTE We could permute the entire matrix, but we might not check every row of the permuted matrix, so this would be inefficient"
        "Also, this doesn't do it the correct way apparently"
        #print(char_matrix[permutation])

        "Make a copy of the entered matrix"
        char_matrix_copy = char_matrix.copy()

        counter = 0
        while len(np.nonzero(sign_matrix[i])) < length:
            "Select the row in which we want to check for nonzero entries"
            column_to_check       = char_matrix_copy[permutation == counter][0]
            print("Column to check is",column_to_check)

            "Find the indices that are nonzero"
            check              = np.nonzero(column_to_check)[0]
            print("The nonzero indices are at",check)

            "If there are nonzero entries, clear the matrix in that column to prevent recounting"
            if len(check) > 0:
                for column in check:
                    char_matrix_copy[:,column] = np.zeros_like(char_matrix_copy[:,column])

            "Update the signature matrix"
            sign_matrix[i][check] = counter + 1
            print("The updated signature matrix is therefore",sign_matrix)    
            
            "Increase the counter by one"
            counter += 1

            if counter > 50:
                print("Manual breaking necessary!")
                break

    print("Resulting signature matrix is",sign_matrix)
    "Reshaping the temporary testing array"
    test_array = test_array.reshape(n_permutations,char_matrix.shape[0])

    "Plotting a histogram of the different permutations of the first index to see if this is actually uniformly distributed"
    # plt.figure()
    # plt.hist(test_array[:,0],color='blue',edgecolor='black')
    # plt.show()

    return None


def permutation_test(n_permutations,seed=seed,index=0):
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
# minhash_slow(char_matrix_test,6)