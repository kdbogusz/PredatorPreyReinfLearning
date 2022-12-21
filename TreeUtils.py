from deap import gp
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from Utils import plot_tree

flow_operators = ["sequence3", "sequence2", "selector2", "selector3"]

def read_checkpoint(filename):
     with open(filename, "rb") as cp_file:
         cp = pickle.load(cp_file)
     return cp

def save_checkpoint(filename, object):
    with open(filename, 'wb') as cp_file:
        pickle.dump(object, cp_file, protocol=pickle.HIGHEST_PROTOCOL)

def remove_node_recursively(node, G, edges):
    for edge in edges:
        if edge[0] == node:
            child = edge[1]
            remove_node_recursively(child, G, edges)
    if node in G.nodes():
        G.remove_node(node)

def simplify_tree(root, G, edges, labels, actions, conditions):
    child_action = None
    child_to_action = {}
    for edge in edges:
        if edge[0] == root:
            child = edge[1]
            if labels[child] in flow_operators:
                _, new_child_action = simplify_tree(child, G, edges, labels, actions, conditions)
                if new_child_action: child_action = new_child_action
                child_to_action[child] = new_child_action
    label = labels[root]
    nodes_to_remove = []
    action = None
    if label.startswith("selector"):
        for edge in edges:
            if edge[0] == root:
                child = edge[1]
                if labels[child] in actions and action is None:
                    action = child
                elif labels[child] in flow_operators and child_to_action[child]:
                    continue
                else:
                    nodes_to_remove.append(child)       
    elif label.startswith("sequence"):
        found_action = False 
        for edge in edges:
            if edge[0] == root:
                child = edge[1]
                if found_action:
                    nodes_to_remove.append(child)
                elif labels[child] in actions:
                    action = child
                    found_action = True
    for node in nodes_to_remove:
        remove_node_recursively(node, G, edges)
    if action is None and child_action is None: 
        remove_node_recursively(root, G, edges)
    return G, (action or child_action)

def prune_tree(checkpoint_path, actions, conditions):
    hof = read_checkpoint(checkpoint_path)
    nodes, edges, labels = gp.graph(hof[0])
    plot_tree(nodes, edges, labels)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    G, _ = simplify_tree(0, G, edges, labels, actions, conditions)
    new_labels = {id: label for id, label in labels.items() if id in G.nodes()}
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, new_labels)
    plt.show()
    
