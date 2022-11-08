import numpy as np
from deap import gp, creator, base, tools, algorithms
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt

from Simulation import run_simulation

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

def sequence3(input1, input2, input3):
        for input in [input1, input2, input3]:
            if input == False:
                return False
            elif input == True:
                continue
            else:
                return input
        return True

def sequence2(input1, input2):
    for input in [input1, input2]:
        if input == False:
            return False
        elif input == True:
            continue
        else:
            return input
    return True

def selector2(input1, input2):
    for input in [input1, input2]:
        if input == False or input == True:
            continue
        else:
            return input
    return 'do_nothing'

def selector3(input1, input2, input3):
    for input in [input1, input2, input3]:
        if input == False or input == True:
            continue
        else:
            return input
    return 'do_nothing'

if __name__ == '__main__':
    pset = gp.PrimitiveSet("main", 2)

    pset.addPrimitive(sequence2, 2)
    pset.addPrimitive(sequence3, 3)
    pset.addPrimitive(selector2, 2)
    pset.addPrimitive(selector3, 3)

    pset.renameArguments(ARG0="food_nearby")
    pset.renameArguments(ARG1="predator_nearby")
    pset.addTerminal('go_to_food')
    pset.addTerminal('go_from_predator')
    pset.addTerminal('do_nothing')

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=3)

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    def eval_prey(individual):
        routine = gp.compile(individual, pset)

        res = run_simulation(routine)

        return res,


    toolbox.register("evaluate", eval_prey)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    _, logbook = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 50, stats, halloffame=hof)
    plot_logbook(logbook)
    nodes,edges,labels = gp.graph(hof[0])
    plot_tree(nodes, edges, labels)

   