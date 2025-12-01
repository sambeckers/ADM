import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"]
})
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
    
    {'seed': 123, 'b': 20, 'r': 5, 'output': 'results_b20_r5_seed123.txt'},
    {'seed': 999, 'b': 20, 'r': 5, 'output': 'results_b20_r5_seed999.txt'},
]

result_files = [
    {'file': 'results_b10_r10.txt', 'label': 'b=10, r=10 (h=100)', 'style': '-', 'color': 'C0'},
    {'file': 'results_b20_r5.txt', 'label': 'b=20, r=5 (h=100)', 'style': '-', 'color': 'C0'},
    {'file': 'results_b25_r4.txt', 'label': 'b=25, r=4 (h=100)', 'style': '-', 'color': 'C0'},
    {'file': 'results_b30_r5.txt', 'label': 'b=30, r=5 (h=150)', 'style': '-', 'color': 'C0'},
    {'file': 'results_b33_r3.txt', 'label': 'b=33, r=3 (h=99)', 'style': '-', 'color': 'C0'},
    {'file': 'results_b50_r2.txt', 'label': 'b=50, r=2 (h=100)', 'style': '-', 'color': 'C0'},
    {'file': 'results_b100_r1.txt', 'label': 'b=100, r=1 (h=100)', 'style': '-', 'color': 'C0'},
    
    {'file': 'results_b20_r3.txt', 'label': 'b=20, r=3 (h=60)', 'style': '--', 'color': 'C1'},
    {'file': 'results_b20_r10.txt', 'label': 'b=20, r=10 (h=200)', 'style': '--', 'color': 'C1'},
    
    {'file': 'results_b10_r5_h50.txt', 'label': 'b=10, r=5 (h=50)', 'style': ':', 'color': 'C2'},
    {'file': 'results_b20_r10_h200.txt', 'label': 'b=20, r=10 (h=200)', 'style': ':', 'color': 'C2'},
    
    {'file': 'results_b20_r5_seed123.txt', 'label': 'seed=123', 'style': '-.', 'color': 'C3'},
    {'file': 'results_b20_r5_seed999.txt', 'label': 'seed=999', 'style': '-.', 'color': 'C3'},
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

    print('\n' + '='*60)
    print('All experiments completed')
    print('='*60)

def plot_s_curves():
    fig, ax = plt.subplots(dpi=300)
    
    s = np.linspace(0, 1, 1000)
    
    configs = [
        {'b': 10, 'r': 10, 'label': 'b=10, r=10', 'file': 'results_b10_r10.txt'},
        {'b': 20, 'r': 5, 'label': 'b=20, r=5', 'file': 'results_b20_r5.txt'},
        {'b': 25, 'r': 4, 'label': 'b=25, r=4', 'file': 'results_b25_r4.txt'},
        {'b': 30, 'r': 5, 'label': 'b=30, r=5', 'file': 'results_b30_r5.txt'},
        {'b': 33, 'r': 3, 'label': 'b=33, r=3', 'file': 'results_b33_r3.txt'},
        {'b': 50, 'r': 2, 'label': 'b=50, r=2', 'file': 'results_b50_r2.txt'},
        {'b': 100, 'r': 1, 'label': 'b=100, r=1', 'file': 'results_b100_r1.txt'},
        {'b': 20, 'r': 3, 'label': 'b=20, r=3', 'file': 'results_b20_r3.txt'},
        {'b': 20, 'r': 10, 'label': 'b=20, r=10', 'file': 'results_b20_r10.txt'},
        {'b': 10, 'r': 5, 'label': 'b=10, r=5', 'file': 'results_b10_r5_h50.txt'},
    ]
    
    for config in configs:
        if os.path.exists(config['file']):
            data = np.loadtxt(config['file'], delimiter=',')
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            if data.shape[1] >= 3:
                config['n_pairs'] = len(data[:, 2])
        else:
            config['n_pairs'] = 0
    
    configs_sorted = sorted(configs, key=lambda x: x['n_pairs'])
    
    cmap = plt.cm.YlOrRd
    colors = [cmap(0.2 + 0.75 * i / (len(configs_sorted) - 1)) for i in range(len(configs_sorted))]
    
    lines = []
    for i, config in enumerate(configs_sorted):
        b, r = config['b'], config['r']
        prob_detection = 1 - (1 - s**r)**b
        line, = ax.plot(s, prob_detection, linewidth=2.5, color=colors[i], 
                label='{} (n={})'.format(config['label'], config['n_pairs']))
        lines.append(line)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(c['n_pairs'] for c in configs_sorted), 
                                                               vmax=max(c['n_pairs'] for c in configs_sorted)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Number of pairs found', fontsize=11)
    
    ax.set_xlabel('Similarity $s$', fontsize=12)
    ax.set_ylabel('Probability of detection', fontsize=12)
    ax.set_title('Probability of detection: $1-(1-s^r)^b$', fontsize=14)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2.5, alpha=0.7, label='$t = 0.5$')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('s_curves.png', dpi=300)
    print('\nS-curve plot saved as s_curves.png')

def plot_similarity_comparison(results, title, filename):
    plt.figure(dpi=300)
    for result in results:
        if os.path.exists(result['file']):
            data = np.loadtxt(result['file'], delimiter=',')
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            if data.shape[1] >= 3:
                similarities = data[:, 2]
                similarities_sorted = np.sort(similarities)
                n_pairs = np.arange(1, len(similarities_sorted) + 1)
                plt.plot(n_pairs, similarities_sorted, marker='o', markersize=2, 
                        label=result['label'], linewidth=1.5)
    plt.xlabel('Number of most similar pairs', fontsize=12)
    plt.ylabel('Jaccard similarity', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print('Plot saved as {}'.format(filename))

def plot_results():
    fixed_h = [
        {'file': 'results_b10_r10.txt', 'label': 'b=10, r=10 (h=100)'},
        {'file': 'results_b20_r5.txt', 'label': 'b=20, r=5 (h=100)'},
        {'file': 'results_b25_r4.txt', 'label': 'b=25, r=4 (h=100)'},
        {'file': 'results_b33_r3.txt', 'label': 'b=33, r=3 (h=99)'},
        {'file': 'results_b50_r2.txt', 'label': 'b=50, r=2 (h=100)'},
        {'file': 'results_b100_r1.txt', 'label': 'b=100, r=1 (h=100)'},
    ]
    
    vary_r = [
        {'file': 'results_b20_r3.txt', 'label': 'b=20, r=3 (h=60)'},
        {'file': 'results_b20_r5.txt', 'label': 'b=20, r=5 (h=100)'},
        {'file': 'results_b20_r10.txt', 'label': 'b=20, r=10 (h=200)'},
    ]
    
    vary_h = [
        {'file': 'results_b10_r5_h50.txt', 'label': 'b=10, r=5 (h=50)'},
        {'file': 'results_b20_r5.txt', 'label': 'b=20, r=5 (h=100)'},
        {'file': 'results_b20_r10_h200.txt', 'label': 'b=20, r=10 (h=200)'},
    ]
    
    vary_seed = [
        {'file': 'results_b20_r5.txt', 'label': 'seed=42'},
        {'file': 'results_b20_r5_seed123.txt', 'label': 'seed=123'},
        {'file': 'results_b20_r5_seed999.txt', 'label': 'seed=999'},
    ]
    
    print('\nPlot saved as effect_of_b.png')
    plot_similarity_comparison(fixed_h, 'Effect of $b$ (bands) with fixed $h\\approx 100$', 'effect_of_b.png')
    plot_similarity_comparison(vary_r, 'Effect of r (rows per band) with b=20', 'effect_of_r.png')
    plot_similarity_comparison(vary_h, 'Effect of h (signature length)', 'effect_of_h.png')
    plot_similarity_comparison(vary_seed, 'Effect of random seed with b=20, r=5', 'effect_of_seed.png')

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


