import numpy as np
import matplotlib.pyplot as plt
import os

result_files = [
    # Effect of varying b (number of bands)
    {'file': 'results_b10_r5_n100.txt', 'label': 'b=10, r=5 (few bands)', 'style': '-'},
    {'file': 'results_b20_r5_n100.txt', 'label': 'b=20, r=5 (medium)', 'style': '-'},
    {'file': 'results_b25_r4_n100.txt', 'label': 'b=25, r=4', 'style': '-'},
    {'file': 'results_b33_r3_n100.txt', 'label': 'b=33, r=3 (many bands)', 'style': '-'},
    {'file': 'results_b50_r2_n100.txt', 'label': 'b=50, r=2 (high precision)', 'style': '-'},
    
    # Effect of more permutations
    {'file': 'results_b20_r5_n150.txt', 'label': 'b=20, r=5, n=150', 'style': '--'},
    {'file': 'results_b20_r5_n200.txt', 'label': 'b=20, r=5, n=200', 'style': '--'},
    
    # Different seeds
    {'file': 'results_b20_r5_n100_seed123.txt', 'label': 'seed=123', 'style': ':'},
    {'file': 'results_b20_r5_n100_seed999.txt', 'label': 'seed=999', 'style': ':'},
]

plt.figure(figsize=(10, 6))

for result in result_files:
    if os.path.exists(result['file']):
        data = np.loadtxt(result['file'], delimiter=',')
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        if data.shape[1] >= 3:
            similarities = data[:, 2]
            similarities_sorted = np.sort(similarities)
            n_pairs = np.arange(1, len(similarities_sorted) + 1)
            
            linestyle = result.get('style', '-')
            plt.plot(n_pairs, similarities_sorted, linestyle=linestyle, marker='o', markersize=2, 
                    label=result['label'], linewidth=1.5)

plt.xlabel('Number of Most Similar Pairs', fontsize=12)
plt.ylabel('Jaccard Similarity', fontsize=12)
plt.title('LSH Performance: Similar Pairs vs Jaccard Similarity', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('similarity_comparison.png', dpi=300)
print('Plot saved as similarity_comparison.png')
plt.show()
