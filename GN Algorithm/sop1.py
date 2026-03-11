import networkx as nx
import matplotlib.pyplot as plt
import time

# Define increasing network sizes
sizes = [10, 30, 50, 100, 150, 200]
gn_times = []
louvain_times = []

print("Running simulations. This may take a moment for larger sizes...")

for n in sizes:
    # Generate a sparse random graph for each size
    G = nx.barabasi_albert_graph(n, 2)
    
    # Measure Girvan-Newman execution time
    start_time = time.time()
    # Using tuple() exhausts the generator to find all communities
    tuple(nx.community.girvan_newman(G)) 
    gn_times.append(time.time() - start_time)
    
    # Measure Louvain execution time for comparison
    start_time = time.time()
    nx.community.louvain_communities(G)
    louvain_times.append(time.time() - start_time)

# Plotting the data
plt.figure(figsize=(8, 5))
plt.plot(sizes, gn_times, label='Girvan-Newman O(n^3)', marker='o', color='red', linewidth=2)
plt.plot(sizes, louvain_times, label='Louvain (Fast/Approximation)', marker='s', color='blue', linewidth=2)

plt.xlabel('Network Size (Number of Nodes)', fontsize=12)
plt.ylabel('Execution Time (Seconds)', fontsize=12)
plt.title('Figure 1.1: Computational Complexity and Execution Time Scaling', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()