# Strassen Matrix Multiplication

A Python implementation of Strassen's divide-and-conquer matrix multiplication, with experiments to find the crossover point where it beats the standard O(n^3) algorithm and an application to counting triangles in random graphs.

## What's here

| File | Purpose |
| --- | --- |
| `strassen.py` | Command-line implementation. Reads two `d x d` matrices and prints the diagonal of their product. |
| `experiments.py` | (1) Times standard vs. one-level Strassen for `n = 1..100` to locate the crossover dimension `n0`; (2) counts triangles in Erdos-Renyi random graphs `G(1024, p)` via `trace(A^3) / 6` and plots them against the expected `C(n,3) p^3`. |
| `runtime_comparison.py` | Plots measured runtime of standard vs. Strassen for `n = 2..256` against the theoretical `n^3` and `n^2.807` growth curves. |

## Approach

- Each recursion splits both matrices into quadrants and computes **7** sub-products instead of 8, giving O(n^log2(7)) ~ O(n^2.81).
- Odd dimensions are handled by zero-padding to the next even size at each level.
- Below a tuned threshold `n0` the recursion falls back to the standard triple loop (`i, k, j` order for cache friendliness). Experiments found different crossover points for even and odd sizes, so the default `n0` is 10 for even `n` and 37 for odd `n` (odd sizes pay the extra padding cost).

## Usage

Requires Python 3 with `numpy` (plus `matplotlib` and `scipy` for the experiments).

```sh
pip install numpy matplotlib scipy

# input file: 2*d*d integers, one per line (matrix A row-major, then matrix B)
python strassen.py 0 <dimension> <inputfile>

python experiments.py
python runtime_comparison.py
```
