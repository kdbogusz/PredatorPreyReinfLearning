import numpy as np
from deap import gp, creator, base, tools, algorithms
import networkx as nx
import matplotlib.pyplot as plt
from FlowOperators import sequence2, sequence3, selector2, selector3, TreeNode, carry


def plot_logbook(logbook):
    min_values = logbook.select("min")
    max_values = logbook.select("max")
    avg_values = logbook.select("avg")
    std_values = logbook.select("std")
    epoch_values = np.arange(len(avg_values))
    plt.errorbar(epoch_values, avg_values, std_values, label="avg +- std", ls='none', capsize=3, fmt='o')
    plt.plot(epoch_values, min_values, "-o", label="min")
    plt.plot(epoch_values, max_values, "-o", label="max")
    plt.legend()
    plt.show()

def plot_tree(nodes, edges, labels):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_pydot.graphviz_layout(g, prog="dot")
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()

def create_pset(terminals, args):
    pset = gp.PrimitiveSetTyped("main", [bool, bool, bool, bool, bool], TreeNode)
    pset.addPrimitive(sequence2, [bool, TreeNode], TreeNode)
    pset.addPrimitive(sequence3, [bool, bool, TreeNode], TreeNode)
    pset.addPrimitive(selector2, [TreeNode, TreeNode], TreeNode)
    pset.addPrimitive(selector3, [TreeNode, TreeNode, TreeNode], TreeNode)
    pset.addPrimitive(carry, [bool], bool)
    deap_args = [f"ARG{i}" for i in range(len(args))]
    kargs = {k: v for k, v in zip(deap_args, args)}
    pset.renameArguments(**kargs)
    for terminal in terminals:
        pset.addTerminal(TreeNode(action=terminal, success=True), TreeNode)
    return pset

def create_toolbox(pset, pool, eval_fn):
    toolbox = base.Toolbox()
    toolbox.register("map", pool.map)

    toolbox.register(f"expr_init", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register(f"individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register(f"population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_fn)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    return toolbox

def create_stats():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats