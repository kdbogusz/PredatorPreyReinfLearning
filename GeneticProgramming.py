from deap import gp, creator, base, tools, algorithms
from Simulation import run_simulation
import multiprocessing
from Utils import create_pset, create_stats, create_toolbox, plot_logbook, plot_tree

PREY_TERMINALS = [
    'go_to_food', 
    'go_from_predator', 
    'do_nothing', 
    'eat', 
    'reproduce'
]
PREY_ARGS = [
    "food_nearby", 
    "predator_nearby", 
    "hunger_over_half", 
    "over_reproduction_age", 
    "on_grass"
]
pset_prey = create_pset(PREY_TERMINALS, PREY_ARGS)

PREDATOR_TERMINALS = [
    'go_to_prey', 
    'do_nothing', 
    'eat', 
    'reproduce'
]
PREDATOR_ARGS = [
    "prey_nearby", 
    "hunger_over_half", 
    "over_reproduction_age", 
    "caught_prey"
]
pset_predator = create_pset(PREDATOR_TERMINALS, PREDATOR_ARGS)

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

def show_behaviour():
    prey_routine = gp.compile(best_prey, pset_prey)
    predator_routine = gp.compile(best_predator, pset_predator)
    run_simulation(prey_routine, predator_routine, draw_grid=True)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

if __name__ == '__main__':

    prey_logs = []
    predator_logs = []
    cpu_count = multiprocessing.cpu_count()
    pool1 = multiprocessing.Pool(cpu_count)
    pool2 = multiprocessing.Pool(cpu_count // 2)

    toolbox_prey = create_toolbox(pset_prey, pool1, eval_prey)
    pop_prey = toolbox_prey.population(n=10)
    hof_prey = tools.HallOfFame(1)
    stats_prey = create_stats()
    _, logbook = algorithms.eaSimple(pop_prey, toolbox_prey, 0.5, 0.2, 5, stats_prey, halloffame=hof_prey)
    best_prey = hof_prey[0]
    nodes, edges, labels = gp.graph(hof_prey[0])
    plot_tree(nodes, edges, labels)

    toolbox_predator = create_toolbox(pset_predator, pool1, eval_predator)
    pop_predator = toolbox_predator.population(n=50)
    hof_predator = tools.HallOfFame(1)
    stats_predator = create_stats()
    _, logbook = algorithms.eaSimple(pop_predator, toolbox_predator, 0.5, 0.3, 10, stats_predator, halloffame=hof_predator)
    best_predator = hof_predator[0]
    nodes, edges, labels = gp.graph(hof_predator[0])
    plot_tree(nodes, edges, labels)
    show_behaviour()


    # for i in range(3):
    #     toolbox_prey = create_toolbox(pset_prey, pool1, eval_prey)
    #     if pop_prey is None: 
    #         pop_prey = toolbox_prey.population(n=10)
    #     hof_prey = tools.HallOfFame(1)
    #     stats_prey = create_stats()

    #     toolbox_predator = create_toolbox(pset_predator, pool2, eval_predator)
    #     if pop_predator is None: 
    #         pop_predator = toolbox_predator.population(n=60)
    #     hof_predator = tools.HallOfFame(1)
    #     stats_predator = create_stats()

    #     if best_prey is None:
    #         _, logbook = algorithms.eaSimple(pop_prey, toolbox_prey, 0.5, 0.2, 5, stats_prey, halloffame=hof_prey)
    #         best_prey = hof_prey[0]
    #         nodes, edges, labels = gp.graph(hof_prey[0])
    #         plot_tree(nodes, edges, labels)

    #     _, logbook = algorithms.eaSimple(pop_predator, toolbox_predator, 0.5, 0.3, 10, stats_predator, halloffame=hof_predator)
    #     best_predator = hof_predator[0]
    #     nodes, edges, labels = gp.graph(hof_predator[0])
    #     plot_tree(nodes, edges, labels)

    #     show_behaviour()
