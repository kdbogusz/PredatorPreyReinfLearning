import numpy as np
import itertools
from GrassAgent import Grass

def von_neumann_neighborhood(n):
     neighborhood = [[(n,i), (-n,i), (i,n), (i,-n)] for i in range(-n,n+1)]
     return sorted(list(set(list(itertools.chain(*neighborhood)))))

class Prey:
    ptype = -1  # 1 if predator, -1 for prey
    age = 0
    epsilon = 0.2

    def __init__(self, x_position, y_position, ID, lastAte, father, reproduction_age,
                 death_rate, reproduction_rate, weights, learning_rate, discount_factor, hunger_minimum,
                 tree_function):

        self.x_position = x_position
        self.y_position = y_position
        self.ID = ID
        self.lastAte = lastAte
        self.father = father
        self.reproduction_age = reproduction_age
        self.death_rate = death_rate
        self.reproduction_rate = reproduction_rate
        self.weights = weights
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.hunger_minimum = hunger_minimum
        self.q = 0
        self.tree_function = tree_function
    
    def in_grid(self, matrix, location):
         return  -1 < location[0] < matrix.xDim and -1 < location[1] < matrix.yDim

    def predator_distance(self, matrix, location):
        for r in range(1,4):
            for dx, dy in von_neumann_neighborhood(r):
                new_location = [location[0] + dx, location[1] + dy]
                if self.in_grid(matrix, new_location):
                    for entity in matrix.grid[new_location[0]][new_location[1]]:
                        if entity.ptype == 1:
                            return r
        return matrix.xDim

    def pick_action(self, matrix, print_move):
        """
        Perform action (i.e. movement) of the agent depending on its evaluations
        """
        grass_nearby = False
        grass_location = None
        furthest_from_predator_location = None
        own_location = np.array([self.x_position, self.y_position])
        location_predator_min_distance = self.predator_distance(matrix, own_location)
        furthest_from_predator_location = own_location
        on_grass = False
        for entity in matrix.grid[own_location[0]][own_location[1]]:
            if entity.ptype == 0:
                on_grass = True 
                break
        for dx, dy in von_neumann_neighborhood(1):
            new_location = [self.x_position + dx, self.y_position + dy]
            if self.in_grid(matrix, new_location):
                for entity in matrix.grid[new_location[0]][new_location[1]]:
                    if entity.ptype == 0:
                        grass_nearby = True
                        grass_location = np.array(new_location)
                location_predator_distance = self.predator_distance(matrix, new_location)
                if location_predator_distance < location_predator_min_distance:
                    location_predator_min_distance = location_predator_distance
                    furthest_from_predator_location = new_location
        result = self.tree_function(on_grass, grass_nearby, location_predator_min_distance < 4, self.lastAte > (self.hunger_minimum // 2), self.age >= self.reproduction_age)
        if print_move:
            print(result)
        if result == 'go_from_predator':
            return furthest_from_predator_location if furthest_from_predator_location is not None else own_location, -1, 0
        if result == 'go_to_food':
            return grass_location if grass_location is not None else own_location, -1, 0
        if result == "eat":
            return own_location, self.Eat(matrix.grid[self.x_position][self.y_position]), 0
        if result == "reproduce":
            return own_location, -1, self.Reproduce()
        return own_location, -1, 0

    def Aging(self, i):
        self.age += 1
        self.epsilon = 1 / i
        if i <= 501:
            self.learning_rate = 0.05 - 0.0001 * (i - 1)
        else:
            self.learning_rate = 0
        self.lastAte += 1
        return

    def Eat(self, agentListAtMatrixPos):
        for agent in agentListAtMatrixPos:  # Not selected randomly at the moment, just eats the first prey in the list
            if type(agent) is Grass:
                killFoodSource = agent.consume()
                self.lastAte = 0
                if killFoodSource == 0:
                    return agent.ID
        return -1

    def Starve(self):
        if self.lastAte > self.hunger_minimum:
            pdeath = self.lastAte * self.death_rate
            r = np.random.rand()
            if r < pdeath:
                return self.ID
        return -1

    def Reproduce(self):
        offspring = 0
        food_in_stomach = self.hunger_minimum - self.lastAte
        offspring_food = food_in_stomach // 2
        if self.age >= self.reproduction_age and self.lastAte < (self.hunger_minimum / 2):
            self.lastAte = self.hunger_minimum - food_in_stomach + offspring_food
            offspring = Prey(self.x_position, self.y_position, -1, self.hunger_minimum - offspring_food, self.ID, self.reproduction_age,
                             self.death_rate, self.reproduction_rate, self.weights, self.learning_rate,
                             self.discount_factor, offspring_food, self.tree_function)  # ID is changed in Grid.update()
        return offspring


class Predator:
    ptype = 1  # 1 if predator, -1 for prey
    age = 0
    epsilon = 0.2

    def __init__(self, x_position, y_position, ID, lastAte, father, reproduction_age,
                 death_rate, reproduction_rate, weights, learning_rate, discount_factor, hunger_minimum,
                 tree_function):

        self.x_position = x_position
        self.y_position = y_position
        self.ID = ID
        self.lastAte = lastAte
        self.father = father
        self.reproduction_age = reproduction_age
        self.death_rate = death_rate
        self.reproduction_rate = reproduction_rate
        self.weights = weights
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.hunger_minimum = hunger_minimum
        self.q = 0
        self.tree_function = tree_function

    def pick_action(self, matrix, print_move):
        """
        Perform action (i.e. movement) of the agent depending on its evaluations
        """

        prey_nearby = False
        prey_location = None
        own_location = np.array([self.x_position, self.y_position])
        for entity in matrix.grid[own_location[0]][own_location[1]]:
            if entity.ptype == -1:
                prey_nearby = True
                prey_location = own_location.copy()
                break
        if prey_location is None:
            for r in range(1,10):
                for dx, dy in von_neumann_neighborhood(r):
                    new_location = [own_location[0] + dx, own_location[1] + dy]
                    if -1 < new_location[0] < matrix.xDim and -1 < new_location[1] < matrix.yDim:
                        for entity in matrix.grid[new_location[0]][new_location[1]]:
                            if entity.ptype == -1:
                                if r < 3:
                                    prey_nearby = True
                                prey_location = new_location.copy()
                                break
                if prey_location: 
                    break

        if self.tree_function is None:
            return own_location, -1, 0

        result = self.tree_function(prey_nearby,
                                    self.lastAte > (self.hunger_minimum // 2),
                                    self.age >= self.reproduction_age,
                                    prey_location is not None and prey_location[0] == own_location[0] and prey_location[1] == own_location[1],
                )
        if print_move:
            print(result)
        if result == 'go_to_prey':
            # print(prey_location)
            if prey_location is None:
                return own_location, -1, 0
            if prey_location[0] == own_location[0] and prey_location[1] == own_location[1]:
                result = "eat"
            else:
                x = prey_location[0] - own_location[0]
                x = x if x == 0 else x // abs(x)
                y = prey_location[1] - own_location[1]
                y = y if y == 0 else y // abs(y)
                return own_location + np.array([x, y]), -1, 0
        if result == "eat":
            return own_location, self.Eat(matrix.grid[self.x_position][self.y_position]), 0
        if result == "reproduce":
            return own_location, -1, self.Reproduce()
        return own_location, -1, 0

    def Aging(self, i):
        self.age += 1
        self.lastAte += 1
        return

    def Eat(self, agentListAtMatrixPos):
        for agent in agentListAtMatrixPos:
            # print(type(agent))
            if type(agent) is Prey:  # Not selected randomly at the moment, just eats the first prey in the list
                # print(agent.ID)
                self.lastAte = 0
                return agent.ID
        return -1

    def Starve(self):
        if self.lastAte > self.hunger_minimum:
            pdeath = self.lastAte * self.death_rate
            r = np.random.rand()
            if r < pdeath:
                return self.ID
        return -1

    def Reproduce(self):
        offspring = 0
        food_in_stomach = self.hunger_minimum - self.lastAte
        offspring_food = food_in_stomach // 2
        if self.age >= self.reproduction_age and self.lastAte < (self.hunger_minimum / 2):
            self.lastAte = self.hunger_minimum - food_in_stomach + offspring_food
            offspring = Predator(self.x_position, self.y_position, -1, self.hunger_minimum - offspring_food, self.ID, self.reproduction_age,
                                 self.death_rate, self.reproduction_rate, self.weights, self.learning_rate,
                                 self.discount_factor, self.hunger_minimum, self.tree_function)  # ID is changed in Grid.update()
        return offspring
