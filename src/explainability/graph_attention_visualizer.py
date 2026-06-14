"""Visualize wallet network topology with GAT attention/node importance."""
import networkx as nx
import matplotlib.pyplot as plt

def save_attention_network(edge_list, node_scores, output_path):
    g = nx.Graph()
    g.add_edges_from(edge_list)
    scores = [node_scores.get(n, 0.1) for n in g.nodes]
    pos = nx.spring_layout(g, seed=42)
    nx.draw_networkx_edges(g, pos, alpha=0.35)
    nx.draw_networkx_nodes(g, pos, node_size=[3000*s + 80 for s in scores])
    nx.draw_networkx_labels(g, pos, font_size=8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
