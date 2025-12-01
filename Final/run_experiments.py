import subprocess
import os

experiments = [
    # Vary b with fixed r=5, n=100 - shows effect of number of bands
    {'seed': 42, 'b': 10, 'r': 5, 'n': 100, 'output': 'results_b10_r5_n100.txt'},   # Few bands = high recall
    {'seed': 42, 'b': 20, 'r': 5, 'n': 100, 'output': 'results_b20_r5_n100.txt'},   # Medium bands
    {'seed': 42, 'b': 25, 'r': 4, 'n': 100, 'output': 'results_b25_r4_n100.txt'},   # More bands
    
    # Good configurations with r<=5 for consistent hashing
    {'seed': 42, 'b': 33, 'r': 3, 'n': 100, 'output': 'results_b33_r3_n100.txt'},   # Your known good config
    {'seed': 42, 'b': 50, 'r': 2, 'n': 100, 'output': 'results_b50_r2_n100.txt'},   # High precision
    
    # Effect of more permutations with fixed b,r
    {'seed': 42, 'b': 20, 'r': 5, 'n': 150, 'output': 'results_b20_r5_n150.txt'},   # Extra permutations
    {'seed': 42, 'b': 20, 'r': 5, 'n': 200, 'output': 'results_b20_r5_n200.txt'},   # More permutations
    
    # Different seeds - shows algorithm stability
    {'seed': 123, 'b': 20, 'r': 5, 'n': 100, 'output': 'results_b20_r5_n100_seed123.txt'},
    {'seed': 999, 'b': 20, 'r': 5, 'n': 100, 'output': 'results_b20_r5_n100_seed999.txt'},
]

for exp in experiments:
    print('\n' + '='*60)
    print('Running experiment: b={}, r={}, n={}'.format(exp['b'], exp['r'], exp['n']))
    print('='*60)
    
    cmd = [
        'python', 'final_assignment.py',
        '--seed', str(exp['seed']),
        '--b', str(exp['b']),
        '--r', str(exp['r']),
        '--n', str(exp['n']),
        '--threshold', '0.5',
        '--output', exp['output']
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__))
        print('Experiment completed: {}'.format(exp['output']))
    except subprocess.CalledProcessError as e:
        print('Experiment failed: {}'.format(e))
    except KeyboardInterrupt:
        print('\nExperiment interrupted by user')
        break

print('\n' + '='*60)
print('All experiments completed')
print('='*60)
