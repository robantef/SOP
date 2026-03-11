import networkx as nx
import matplotlib.pyplot as plt
import time

# --- Synthetic scaling benchmark (Girvan-Newman only, up to 200 nodes) ---
sizes = [10, 30, 50, 100, 150, 200]
gn_times = []

print("Running Girvan-Newman simulations...")
for n in sizes:
    G = nx.barabasi_albert_graph(n, 2)

    start_time = time.time()
    tuple(nx.community.girvan_newman(G))
    gn_times.append(time.time() - start_time)

    print(f"  n={n:3d}: GN={gn_times[-1]:.3f}s")

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(sizes, gn_times, label='Girvan-Newman O(n\u00b3)',
         marker='o', color='red', linewidth=2)

plt.xlabel('Network Size (Number of Nodes)', fontsize=12)
plt.ylabel('Execution Time (Seconds)', fontsize=12)
plt.title('Figure 1.1: Computational Complexity and Execution Time Scaling', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('figure_1_1_complexity.png', dpi=150, bbox_inches='tight')
plt.show()