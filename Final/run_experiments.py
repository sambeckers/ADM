import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

experiments = [
    {'seed': 42, 'b': 10, 'r': 10, 'output': 'results_b10_r10.txt'},
    {'seed': 42, 'b': 20, 'r': 5, 'output': 'results_b20_r5.txt'},
    {'seed': 42, 'b': 25, 'r': 4, 'output': 'results_b25_r4.txt'},
    {'seed': 42, 'b': 30, 'r': 5, 'output': 'results_b30_r5.txt'},
    {'seed': 42, 'b': 33, 'r': 3, 'output': 'results_b33_r3.txt'},
    {'seed': 42, 'b': 50, 'r': 2, 'output': 'results_b50_r2.txt'},
    {'seed': 42, 'b': 100, 'r': 1, 'output': 'results_b100_r1.txt'},
    
    {'seed': 42, 'b': 20, 'r': 3, 'output': 'results_b20_r3.txt'},
    {'seed': 42, 'b': 20, 'r': 10, 'output': 'results_b20_r10.txt'},
    
    {'seed': 42, 'b': 10, 'r': 5, 'output': 'results_b10_r5_h50.txt'},
    {'seed': 42, 'b': 20, 'r': 10, 'output': 'results_b20_r10_h200.txt'},
]

result_files = [
    {'file': 'results_b10_r10.txt', 'label': 'b=10, r=10 (h=100)', 'style': '-', 'color': 'C0'},
    {'file': 'results_b20_r5.txt', 'label': 'b=20, r=5 (h=100)', 'style': '-', 'color': 'C0'},
    {'file': 'results_b25_r4.txt', 'label': 'b=25, r=4 (h=100)', 'style': '-', 'color': 'C0'},
    {'file': 'results_b33_r3.txt', 'label': 'b=33, r=3 (h=99)', 'style': '-', 'color': 'C0'},
    {'file': 'results_b50_r2.txt', 'label': 'b=50, r=2 (h=100)', 'style': '-', 'color': 'C0'},
    {'file': 'results_b100_r1.txt', 'label': 'b=100, r=1 (h=100)', 'style': '-', 'color': 'C0'},
    
    {'file': 'results_b20_r3.txt', 'label': 'b=20, r=3 (h=60)', 'style': '--', 'color': 'C1'},
    {'file': 'results_b20_r10.txt', 'label': 'b=20, r=10 (h=200)', 'style': '--', 'color': 'C1'},
    
    {'file': 'results_b10_r5_h50.txt', 'label': 'b=10, r=5 (h=50)', 'style': ':', 'color': 'C2'},
    {'file': 'results_b20_r10_h200.txt', 'label': 'b=20, r=10 (h=200)', 'style': ':', 'color': 'C2'},
]

def run_experiments():
    for exp in experiments:
        if os.path.exists(exp['output']):
            print('\n' + '='*60)
            print('Skipping experiment: b={}, r={} (output file already exists)'.format(exp['b'], exp['r']))
            print('='*60)
            continue
            
        print('\n' + '='*60)
        print('Running experiment: b={}, r={}'.format(exp['b'], exp['r']))
        print('='*60)
        
        cmd = [
            'python', 'final_assignment.py',
            '--seed', str(exp['seed']),
            '--b', str(exp['b']),
            '--r', str(exp['r']),
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

def plot_s_curves():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    s = np.linspace(0, 1, 1000)
    
    configs = [
        {'b': 20, 'r': 5, 'label': 'b=20, r=5'},
        {'b': 30, 'r': 5, 'label': 'b=30, r=5'},
        {'b': 10, 'r': 10, 'label': 'b=10, r=10'},
        {'b': 50, 'r': 2, 'label': 'b=50, r=2'},
        {'b': 100, 'r': 1, 'label': 'b=100, r=1'},
    ]
    
    ax.plot(s, s**5, 'k-', label='$s^r$ (r=5)', linewidth=2)
    ax.plot(s, 1 - s**5, 'k--', label='$1-s^r$ (r=5)', linewidth=2)
    
    for config in configs:
        b, r = config['b'], config['r']
        prob_detection = 1 - (1 - s**r)**b
        ax.plot(s, prob_detection, linewidth=2, label=config['label'])
    
    ax.set_xlabel('Similarity s', fontsize=12)
    ax.set_ylabel('Probability of detection', fontsize=12)
    ax.set_title('LSH S-Curves: Probability of Detection vs Similarity', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('s_curves.png', dpi=300)
    print('\nS-curve plot saved as s_curves.png')
    plt.show()

def plot_results():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    fixed_h = [
        {'file': 'results_b10_r10.txt', 'label': 'b=10, r=10'},
        {'file': 'results_b20_r5.txt', 'label': 'b=20, r=5'},
        {'file': 'results_b25_r4.txt', 'label': 'b=25, r=4'},
        {'file': 'results_b33_r3.txt', 'label': 'b=33, r=3'},
        {'file': 'results_b50_r2.txt', 'label': 'b=50, r=2'},
        {'file': 'results_b100_r1.txt', 'label': 'b=100, r=1'},
    ]
    
    vary_r = [
        {'file': 'results_b20_r3.txt', 'label': 'r=3 (h=60)'},
        {'file': 'results_b20_r5.txt', 'label': 'r=5 (h=100)'},
        {'file': 'results_b20_r10.txt', 'label': 'r=10 (h=200)'},
    ]
    
    vary_h = [
        {'file': 'results_b10_r5_h50.txt', 'label': 'h=50 (b=10)'},
        {'file': 'results_b20_r5.txt', 'label': 'h=100 (b=20)'},
        {'file': 'results_b20_r10_h200.txt', 'label': 'h=200 (b=20)'},
    ]
    
    for result in fixed_h:
        if os.path.exists(result['file']):
            data = np.loadtxt(result['file'], delimiter=',')
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            if data.shape[1] >= 3:
                similarities = data[:, 2]
                n_pairs = len(similarities)
                axes[0].scatter(n_pairs, np.mean(similarities), s=100, label=result['label'])
    
    axes[0].set_xlabel('Number of Pairs Found', fontsize=11)
    axes[0].set_ylabel('Average Jaccard Similarity', fontsize=11)
    axes[0].set_title('Effect of b (bands) with fixed h≈100', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    for result in vary_r:
        if os.path.exists(result['file']):
            data = np.loadtxt(result['file'], delimiter=',')
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            if data.shape[1] >= 3:
                similarities = data[:, 2]
                n_pairs = len(similarities)
                axes[1].scatter(n_pairs, np.mean(similarities), s=100, label=result['label'])
    
    axes[1].set_xlabel('Number of Pairs Found', fontsize=11)
    axes[1].set_ylabel('Average Jaccard Similarity', fontsize=11)
    axes[1].set_title('Effect of r (rows per band) with b=20', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    for result in vary_h:
        if os.path.exists(result['file']):
            data = np.loadtxt(result['file'], delimiter=',')
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            if data.shape[1] >= 3:
                similarities = data[:, 2]
                n_pairs = len(similarities)
                axes[2].scatter(n_pairs, np.mean(similarities), s=100, label=result['label'])
    
    axes[2].set_xlabel('Number of Pairs Found', fontsize=11)
    axes[2].set_ylabel('Average Jaccard Similarity', fontsize=11)
    axes[2].set_title('Effect of h (signature length)', fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_analysis.png', dpi=300)
    print('\nPlot saved as parameter_analysis.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LSH experiments and plot results')
    parser.add_argument('--run', action='store_true', help='Run experiments')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--scurve', action='store_true', help='Plot S-curves')
    
    args = parser.parse_args()
    
    if not args.run and not args.plot and not args.scurve:
        run_experiments()
        plot_results()
        plot_s_curves()
    else:
        if args.run:
            run_experiments()
        if args.plot:
            plot_results()
        if args.scurve:
            plot_s_curves()


