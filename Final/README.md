# Advances in Data Mining: Final Assignment
### Authors: Suzanne van Elten, Bram Bouma & Sam Beckers

## Requirements
- Python
- numpy
- scipy
- matplotlib
- tqdm

Install dependencies:
```bash
pip install numpy scipy matplotlib tqdm
```

## Running the Main Script

Basic usage:
```bash
python final_assignment.py
```

With custom parameters:
```bash
python final_assignment.py --seed 42 --b 20 --r 5 --threshold 0.5 --output results.txt --results_format full
```

Parameters:
- `--seed`: Random seed (default: 42)
- `--b`: Number of bands (default: 20)
- `--r`: Rows per band (default: 5)
- `--n`: Number of permutations (default: auto-calculated as b*r)
- `--threshold`: Similarity threshold (default: 0.5)
- `--output`: Output file name (default: results.txt)
- `--results_format`: Output format - 'full' for u1,u2,similarity or 'pairs' for u1,u2 only (default: full)

Note: The script has a 30-minute timeout. Partial results are saved if timeout occurs.

## Running Experiments

Run all experiments:
```bash
python run_experiments.py
```

Run specific tasks:
```bash
python run_experiments.py --run      # Run experiments only
python run_experiments.py --plot     # Generate plots only
python run_experiments.py --scurve   # Generate S-curve plot only
```

## Output Format

Results are saved as CSV:
- **Full format** (default): `user1,user2,similarity` - includes Jaccard similarity score (useful for plotting)
- **Pairs format**: `user1,user2` - just the user pairs without similarity

In both formats, user1 < user2 (sorted).

Example with pairs format:
```bash
python final_assignment.py --results_format pairs --output results_pairs.txt
```
