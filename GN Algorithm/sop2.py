import warnings
warnings.filterwarnings('ignore')  # Suppress WNTR curve warnings

import wntr
import networkx as nx
import matplotlib.pyplot as plt

# Load the C-Town water network dataset
wn = wntr.network.WaterNetworkModel('CTOWN.INP')
G = nx.Graph(wn.to_graph())
print(f"C-Town loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

leaf_count = sum(1 for n, d in G.degree() if d == 1)
print(f"Leaf nodes (degree-1 dead ends): {leaf_count}")

# Track modularity and minimum community size across GN iterations
MAX_STEPS = 100
modularities, min_sizes, n_comms = [], [], []
singleton_idx = None

print("Tracking GN community evolution (this may take a minute)...")
for i, partition in enumerate(nx.community.girvan_newman(G)):
    mod = nx.community.modularity(G, partition)
    min_sz = min(len(c) for c in partition)

    modularities.append(mod)
    min_sizes.append(min_sz)
    n_comms.append(len(partition))

    if min_sz == 1 and singleton_idx is None:
        singleton_idx = i

    if i % 10 == 0:
        print(f"  Step {i+1:3d}: {len(partition)} communities, "
              f"min_size={min_sz}, Q={mod:.4f}")

    if i >= MAX_STEPS - 1:
        break

peak_idx = modularities.index(max(modularities))
iterations = list(range(1, len(modularities) + 1))

print(f"\nPeak modularity at step {peak_idx + 1}")
if singleton_idx is not None:
    print(f"First singleton at step {singleton_idx + 1}")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Top subplot: Modularity over iterations
ax1.plot(iterations, modularities, color='royalblue', linewidth=2,
         marker='o', markersize=4, label='Modularity score')
ax1.axvline(x=peak_idx + 1, color='green', linestyle='--', linewidth=1.5,
            label=f'Optimal partition (peak Q, step {peak_idx + 1})')
if singleton_idx is not None:
    ax1.axvline(x=singleton_idx + 1, color='red', linestyle='--', linewidth=1.5,
                label=f'First singleton appears (step {singleton_idx + 1})')
ax1.set_ylabel('Modularity (Q)', fontsize=12)
ax1.set_title('Figure 1.2: Premature Fragmentation in Girvan-Newman\n'
              '(C-Town Water Network, 396 Nodes)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# Bottom subplot: Smallest community size over iterations
ax2.plot(iterations, min_sizes, color='crimson', linewidth=2,
         marker='o', markersize=4, label='Smallest community size')
ax2.axhline(y=1, color='darkred', linestyle=':', linewidth=1.2, alpha=0.7,
            label='Singleton threshold (size = 1)')
ax2.axvline(x=peak_idx + 1, color='green', linestyle='--', linewidth=1.5,
            label=f'Optimal partition (step {peak_idx + 1})')
if singleton_idx is not None:
    ax2.axvline(x=singleton_idx + 1, color='red', linestyle='--', linewidth=1.5,
                label=f'First singleton (step {singleton_idx + 1})')
ax2.set_xlabel('Girvan-Newman Iteration (Edge Removals)', fontsize=12)
ax2.set_ylabel('Smallest Community\nSize (Nodes)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figure_1_2_premature_fragmentation.png', dpi=150, bbox_inches='tight')
plt.show()