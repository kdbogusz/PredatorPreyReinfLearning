import numpy as np
from deap import gp, creator, base, tools, algorithms
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

    pset_prey = gp.PrimitiveSet("main", 4)

    pset_prey.addPrimitive(sequence2, 2)
    pset_prey.addPrimitive(sequence3, 3)
    pset_prey.addPrimitive(selector2, 2)
    pset_prey.addPrimitive(selector3, 3)

    pset_prey.renameArguments(ARG0="food_nearby")
    pset_prey.renameArguments(ARG1="predator_nearby")
    pset_prey.renameArguments(ARG2="hunger_over_half")
    pset_prey.renameArguments(ARG3="over_reproduction_age")
    pset_prey.addTerminal('go_to_food')
    pset_prey.addTerminal('go_from_predator')
    pset_prey.addTerminal('do_nothing')
    pset_prey.addTerminal('eat')
    pset_prey.addTerminal('reproduce')

    pset_predator = gp.PrimitiveSet("main", 4)

    pset_predator.addPrimitive(sequence2, 2)
    pset_predator.addPrimitive(sequence3, 3)
    pset_predator.addPrimitive(selector2, 2)
    pset_predator.addPrimitive(selector3, 3)

    pset_predator.renameArguments(ARG0="prey_nearby")
    pset_predator.renameArguments(ARG1="hunger_over_half")
    pset_predator.renameArguments(ARG2="over_reproduction_age")
    pset_predator.renameArguments(ARG3="caught_prey")
    pset_predator.addTerminal('go_to_prey')
    pset_predator.addTerminal('do_nothing')
    pset_predator.addTerminal('eat')
    pset_predator.addTerminal('reproduce')

    best_prey = None
    best_predator = None

    pop_prey = None
    pop_predator = None


    def eval_prey(individual):
        routine = gp.compile(individual, pset_prey)
        predator_routine = gp.compile(best_predator, pset_predator)

        _, res = run_simulation(routine, predator_routine)

        return res,


    def eval_predator(individual):
        routine = gp.compile(individual, pset_predator)
        prey_routine = gp.compile(best_prey, pset_prey)

        res, _ = run_simulation(prey_routine, routine)

        return res,

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    for i in range(10):

        toolbox_prey = base.Toolbox()
        toolbox_predator = base.Toolbox()

        toolbox_prey.register("expr_init_prey", gp.genFull, pset=pset_prey, min_=1, max_=3)
        toolbox_predator.register("expr_init_predator", gp.genFull, pset=pset_predator, min_=1, max_=3)

        toolbox_prey.register("individual_prey", tools.initIterate, creator.Individual, toolbox_prey.expr_init_prey)
        toolbox_predator.register("individual_predator", tools.initIterate, creator.Individual, toolbox_predator.expr_init_predator)

        toolbox_prey.register("population_prey", tools.initRepeat, list, toolbox_prey.individual_prey)
        toolbox_predator.register("population_predator", tools.initRepeat, list, toolbox_predator.individual_predator)

        toolbox_prey.register("evaluate", eval_prey)
        toolbox_prey.register("select", tools.selTournament, tournsize=3)
        toolbox_prey.register("mate", gp.cxOnePoint)
        toolbox_prey.register("expr_mut", gp.genFull, min_=1, max_=3)
        toolbox_prey.register("mutate", gp.mutUniform, expr=toolbox_prey.expr_mut, pset=pset_prey)

        toolbox_predator.register("evaluate", eval_predator)
        toolbox_predator.register("select", tools.selTournament, tournsize=3)
        toolbox_predator.register("mate", gp.cxOnePoint)
        toolbox_predator.register("expr_mut", gp.genFull, min_=1, max_=3)
        toolbox_predator.register("mutate", gp.mutUniform, expr=toolbox_predator.expr_mut, pset=pset_predator)

        if pop_prey is None:
            pop_prey = toolbox_prey.population_prey(n=10)
        hof_prey = tools.HallOfFame(1)
        stats_prey = tools.Statistics(lambda ind: ind.fitness.values)
        stats_prey.register("avg", np.mean)
        stats_prey.register("std", np.std)
        stats_prey.register("min", np.min)
        stats_prey.register("max", np.max)

        if pop_predator is None:
            pop_predator = toolbox_predator.population_predator(n=10)
        hof_predator = tools.HallOfFame(1)
        stats_predator = tools.Statistics(lambda ind: ind.fitness.values)
        stats_predator.register("avg", np.mean)
        stats_predator.register("std", np.std)
        stats_predator.register("min", np.min)
        stats_predator.register("max", np.max)

        _, logbook = algorithms.eaSimple(pop_prey, toolbox_prey, 0.5, 0.2, 5, stats_prey, halloffame=hof_prey)
        best_prey = hof_prey[0]

        _, logbook = algorithms.eaSimple(pop_predator, toolbox_predator, 0.5, 0.2, 5, stats_predator, halloffame=hof_predator)
        best_predator = hof_predator[0]
        nodes, edges, labels = gp.graph(hof_predator[0])
        plot_tree(nodes, edges, labels)
    # plot_logbook(logbook)
    # nodes, edges, labels = gp.graph(hof[0])
    # plot_tree(nodes, edges, labels)

   
