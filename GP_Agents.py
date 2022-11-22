import numpy as np

from GrassAgent import Grass


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

    # def compute_how_many(self, matrix, x=None, y=None):
    #     """
    #     Returns the number of the different agents for each neighbor cell
    #     """
    #     if x is None and y is None:
    #         x = self.x_position
    #         y = self.y_position
    #     how_many = np.zeros([3, 9])  # Grass is nr 0, prey 1 and predator 2.
    #     iMoore = 0
    #     for i in [-1, 0, 1]:
    #         for j in [-1, 0, 1]:
    #             x_target = (x + i) % matrix.xDim
    #             y_target = (y + j) % matrix.yDim
    #
    #             for agents in matrix.grid[x_target][y_target]:
    #
    #                 if type(agents) is Grass:
    #                     how_many[0][iMoore] += 1
    #                 if type(agents) is Prey:
    #                     how_many[1][iMoore] += 1
    #                 if type(agents) is Predator:
    #                     how_many[2][iMoore] += 1
    #             iMoore += 1
    #     return how_many

    # def perceive(self, x, y, matrix):
    #     """
    #     Returns the features for a position x,y as a matrix 9x3
    #     """
    #     # Row 1: grass, Row 2: prey, Row 3: predators
    #     features = np.zeros(12)
    #     # Count all predators and prey in the world
    #     nr_grass = matrix.numGrass
    #     nr_prey = matrix.numPrey
    #     nr_pred = matrix.numPred
    #
    #     # How many agents are at each spot?
    #     how_many = self.compute_how_many(matrix, x, y)
    #
    #     # Calculate features. Note: this is for a prey
    #     features[0] = sum(how_many[2][:]) / nr_pred if nr_pred !=0 else 0
    #     features[1] = sum(how_many[1][:]) / nr_prey if nr_prey != 0 else 0
    #     features[2] = sum(how_many[0][:]) / nr_grass if nr_grass != 0 else 0
    #
    #     predator_neighbors_count = sum(how_many[2][:])
    #     for i in range(8):
    #         features[3+i] = how_many[2][i] / predator_neighbors_count if predator_neighbors_count else 0
    #     return features

    def pick_action(self, matrix, print_move):
        """
        Perform action (i.e. movement) of the agent depending on its evaluations
        """
        grass_nearby = False
        predator_nearby = False
        grass_location = None
        predator_location = None
        furthest_from_predator_location = None
        own_location = np.array([self.x_position, self.y_position])
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if not (dx == 0 and dy == 0) and -1 < self.x_position + dx < matrix.xDim and -1 < self.y_position + dy < matrix.yDim:
                    for entity in matrix.grid[self.x_position + dx][self.y_position + dy]:
                        if entity.ptype == 0:
                            grass_nearby = True
                            grass_location = np.array([self.x_position + dx, self.y_position + dy])
                        if entity.ptype == 1:
                            predator_nearby = True
                            predator_location = np.array([self.x_position + dx, self.y_position + dy])
                            if -1 < self.x_position - dx < matrix.xDim and -1 < self.y_position - dy < matrix.yDim:
                                furthest_from_predator_location = np.array([self.x_position - dx, self.y_position - dy])
                            elif -1 < self.x_position - dx < matrix.xDim:
                                furthest_from_predator_location = np.array([self.x_position - dx, self.y_position])
                            elif -1 < self.y_position - dy < matrix.xDim:
                                furthest_from_predator_location = np.array([self.x_position, self.y_position - dy])
                            else:
                                furthest_from_predator_location = own_location

        if self.tree_function is None:
            return own_location, -1, 0

        # print(self.tree_function)
        result = self.tree_function(grass_nearby, predator_nearby, self.lastAte > (self.hunger_minimum // 2), self.age >= self.reproduction_age)
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

    # ---------------------------Learning part-------------------------------#
    # def Get_Reward(self, matrix):
    #     """
    #     opponent :number of the other species type within the agent’s Moore
    #     neighborhood normalized by the number of total
    #      type is 1 for predator and −1 for prey
    #     same = {0, 1} for if the opponent is on the same location
    #     """
    #     type_animal = self.ptype
    #     how_many = self.compute_how_many(matrix)
    #     x = self.x_position
    #     y = self.y_position
    #     features = self.perceive(x, y, matrix)
    #     feature_wanted = features[0]
    #     opponent = feature_wanted
    #     same = how_many[2][4] > 0
    #     reward = opponent * type_animal + 2 * same * type_animal
    #
    #     return reward

    # def Get_QFunction(self, features):
    #     weights = self.weights
    #
    #     Q = 0
    #     for i in range(len(weights)):
    #         Q = Q + weights[i] * features[i]
    #
    #     return Q

    # def Update_Weight(self, reward, matrix, Q_value):
    #     weights = self.weights
    #     learning_rate = self.learning_rate
    #     discount_factor = self.discount_factor
    #
    #     # Compute the Q'-table:
    #     Q_prime = []
    #     for i in [-1, 0, 1]:
    #         for j in [-1, 0, 1]:
    #             x_target = (self.x_position + i) % matrix.xDim
    #             y_target = (self.y_position + j) % matrix.yDim
    #             features = self.perceive(x_target, y_target, matrix)
    #             Q_prime.append(self.Get_QFunction(features))
    #
    #     # Update the weights:
    #     Q_prime_max = max(Q_prime)
    #     for i in range(0, len(weights)):
    #         if i < 3:
    #             c = 9 / (matrix.xDim * matrix.yDim)
    #         else:
    #             c = 1 / 9
    #
    #         w = weights[i]
    #         f = features[i]
    #         f = np.exp(-0.5 * (f - c) ** 2)
    #
    #         weights[i] = w + learning_rate * (reward + discount_factor * Q_prime_max - Q_value) * f
    #
    #     self.weights = weights
    #     return

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
            offspring = Prey(self.x_position, self.y_position, -1, offspring_food, self.ID, self.reproduction_age,
                             self.death_rate, self.reproduction_rate, self.weights, self.learning_rate,
                             self.discount_factor, self.hunger_minimum, self.tree_function)  # ID is changed in Grid.update()
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

    # def compute_how_many(self, matrix):
    #     """
    #     Returns the number of the different agents for each neighbor cell
    #     """
    #     how_many = np.zeros([3, 9])  # Grass is nr 0, prey 1 and predator 2.
    #     iMoore = 0
    #     for i in [-1, 0, 1]:
    #         for j in [-1, 0, 1]:
    #             x_target = (self.x_position + i) % matrix.xDim
    #             y_target = (self.y_position + j) % matrix.yDim
    #             for agents in matrix.grid[x_target][y_target]:
    #                 if type(agents) is Grass:
    #                     how_many[0][iMoore] += 1
    #                 if type(agents) is Prey:
    #                     how_many[1][iMoore] += 1
    #                 if type(agents) is Predator:
    #                     how_many[2][iMoore] += 1
    #             iMoore += 1
    #     return how_many

    # def perceive(self, x, y, matrix):
    #     """
    #     Returns the features for a position (x,y) as a matrix 9x3
    #     """
    #     # Row 1: grass, Row 2: prey, Row 3: predators
    #     features = np.zeros(12)
    #     # Count all predators and prey in the world
    #     nr_grass = matrix.numGrass
    #     nr_prey = matrix.numPrey
    #     nr_pred = matrix.numPred
    #
    #     # How many agents are at each spot?
    #     how_many = np.zeros([3, 9])  # Grass is nr 0, prey 1 and predator 2.
    #     iMoore = 0
    #     for i in [-1, 0, 1]:
    #         for j in [-1, 0, 1]:
    #             x_target = (x + i) % matrix.xDim
    #             y_target = (y + j) % matrix.yDim
    #
    #             for agents in matrix.grid[x_target][y_target]:
    #
    #                 if type(agents) is Grass:
    #                     how_many[0][iMoore] += 1
    #                 elif type(agents) is Prey:
    #                     how_many[1][iMoore] += 1
    #                 else:
    #                     how_many[2][iMoore] += 1
    #             iMoore += 1
    #     # Calculate features. Note: this is for a predator
    #     if nr_prey != 0:
    #         features[0] = sum(how_many[1][:]) / nr_prey  # prey
    #     else:
    #         features[0] = 0
    #     if nr_pred != 0:
    #         features[1] = sum(how_many[2][:]) / nr_pred  # predators
    #     else:
    #         features[1] = 0
    #     if nr_grass != 0:
    #         features[2] = sum(how_many[0][:]) / nr_grass  # grass
    #     else:
    #         features[2] = 0
    #
    #     if sum(how_many[1][:]) == 0:
    #         features[3:12] = 0
    #     else:
    #         features[3] = how_many[1][0] / sum(how_many[1][:])
    #         features[4] = how_many[1][1] / sum(how_many[1][:])
    #         features[5] = how_many[1][2] / sum(how_many[1][:])
    #         features[6] = how_many[1][3] / sum(how_many[1][:])
    #         features[7] = how_many[1][4] / sum(how_many[1][:])
    #         features[8] = how_many[1][5] / sum(how_many[1][:])
    #         features[9] = how_many[1][6] / sum(how_many[1][:])
    #         features[10] = how_many[1][7] / sum(how_many[1][:])
    #         features[11] = how_many[1][8] / sum(how_many[1][:])
    #
    #     return features

    # def Cells_Evaluation(self, matrix):
    #     """
    #     Evaluate the neighbooring cells
    #     """
    #     x = self.x_position
    #     y = self.y_position
    #     iMoore = 0
    #     score = np.empty([3, 9])
    #     for i_x_Moore in [-1, 0, 1]:
    #         for i_y_Moore in [-1, 0, 1]:
    #             x_eval = np.mod(x + i_x_Moore, matrix.xDim)  # eval case 0 if agent in case 50
    #             y_eval = np.mod(y + i_y_Moore, matrix.yDim)
    #             f_i = self.perceive(x_eval, y_eval, matrix)  # self.perceive or perceive ?
    #             cell_score = np.dot(f_i, self.weights)
    #             score[0][iMoore] = x_eval  # gives the score and the absolute position
    #             score[1][iMoore] = y_eval
    #             score[2][iMoore] = cell_score
    #             iMoore += 1
    #     return score

    def pick_action(self, matrix, print_move):
        """
        Perform action (i.e. movement) of the agent depending on its evaluations
        """

        prey_nearby = False
        prey_location = None
        own_location = np.array([self.x_position, self.y_position])

        for distance in range(max(matrix.xDim, matrix.yDim)):
            if prey_location is not None:
                break

            current_position = np.array([self.x_position - distance, self.y_position - distance])

            while prey_location is None:
                # print(own_location, current_position, distance)
                if -1 < current_position[0] < matrix.xDim and -1 < current_position[1] < matrix.yDim:
                    # print(current_position)
                    for entity in matrix.grid[current_position[0]][current_position[1]]:
                        # print(entity.ptype)
                        if entity.ptype == -1:
                            if distance == 1:
                                prey_nearby = True
                            prey_location = current_position.copy()
                            break

                if distance == 0:
                    break

                if current_position[1] == self.y_position - distance:
                    if current_position[0] == self.x_position + distance:
                        current_position[1] += 1
                    elif current_position[0] < 0:
                        current_position[0] = 0
                    elif current_position[0] >= matrix.xDim:
                        current_position[0] = self.x_position + distance
                    else:
                        current_position[0] += 1
                elif current_position[0] == self.x_position + distance:
                    if current_position[1] == self.y_position + distance:
                        current_position[0] -= 1
                    elif current_position[1] < 0:
                        current_position[1] = 0
                    elif current_position[1] >= matrix.yDim:
                        current_position[1] = self.y_position + distance
                    else:
                        current_position[1] += 1
                elif current_position[1] == self.y_position + distance:
                    if current_position[0] == self.x_position - distance:
                        current_position[1] -= 1
                    elif current_position[0] >= matrix.xDim:
                        current_position[0] = matrix.xDim - 1
                    elif current_position[0] < 0:
                        current_position[0] = self.x_position - distance
                    else:
                        current_position[0] -= 1
                else:
                    if current_position[1] >= matrix.yDim:
                        current_position[1] = matrix.yDim - 1
                    elif current_position[1] < 0:
                        current_position[1] = self.y_position - distance
                    else:
                        current_position[1] -= 1

                if current_position[0] == self.x_position - distance and current_position[1] == self.y_position - distance:
                    break

        if self.tree_function is None:
            return own_location, -1, 0

        # print(prey_location, own_location)
        result = self.tree_function(prey_nearby,
                                    self.lastAte > (self.hunger_minimum // 2),
                                    self.age >= self.reproduction_age,
                                    prey_location is not None and prey_location[0] == own_location[0] and prey_location[1] == own_location[1])
        if print_move:
            print(result)
        if result == 'go_to_prey':
            # print(prey_location)
            if prey_location is None:
                return own_location, -1, 0
            x = prey_location[0] - own_location[0]
            x = x if x == 0 else x // abs(x)
            y = prey_location[1] - own_location[1]
            y = y if y == 0 else y // abs(y)
            # print(x, y)
            # print(prey_location)
            # print(own_location, prey_location, own_location + np.array([x, y]))
            # print(own_location + np.array([x, y]))
            return own_location + np.array([x, y]), -1, 0
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

    # ---------------------------Learning part-------------------------------#
    # def Get_Reward(self, matrix):
    #     """
    #     opponent :number of the other species type within the agent’s Moore
    #     neighborhood normalized by the number of total
    #     type is 1 for predator and −1 for prey
    #     same = {0, 1} for if the opponent is on the same location
    #     """
    #     type_animal = self.ptype
    #     how_many = self.compute_how_many(matrix)
    #     x = self.x_position
    #     y = self.y_position
    #     features = self.perceive(x, y, matrix)
    #     feature_wanted = features[0]
    #     opponent = feature_wanted
    #     same = how_many[1][4] > 0
    #     reward = opponent * type_animal + 2 * same * type_animal
    #     return reward

    # def Get_QFunction(self, features):
    #     weights = self.weights
    #
    #     Q = 0
    #     for i in range(len(weights)):
    #         Q = Q + weights[i] * features[i]
    #
    #     return Q

    # def Update_Weight(self, reward, matrix, Q_value):
    #     weights = self.weights
    #     learning_rate = self.learning_rate
    #     discount_factor = self.discount_factor
    #
    #     # Compute the Q'-table:
    #     Q_prime = []
    #     for i in [-1, 0, 1]:
    #         for j in [-1, 0, 1]:
    #             x_target = (self.x_position + i) % matrix.xDim
    #             y_target = (self.y_position + j) % matrix.yDim
    #
    #             features = self.perceive(x_target, y_target, matrix)
    #             Q_prime.append(self.Get_QFunction(features))
    #
    #     # Update the weights:
    #     Q_prime_max = max(Q_prime)
    #     for i in range(0, len(weights)):
    #         if i < 3:
    #             c = 9 / (matrix.xDim * matrix.yDim)
    #         else:
    #             c = 1 / 9
    #
    #         w = weights[i]
    #         f = features[i]
    #         f = np.exp(-0.5 * (f - c) ** 2)
    #         weights[i] = w + learning_rate * (reward + discount_factor * Q_prime_max - Q_value) * f
    #
    #     self.weights = weights
    #     return

    def Eat(self, agentListAtMatrixPos):
        for agent in agentListAtMatrixPos:
            # print(type(agent))
            if type(agent) is Prey:  # Not selected randomly at the moment, just eats the first prey in the list
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
            offspring = Predator(self.x_position, self.y_position, -1, offspring_food, self.ID, self.reproduction_age,
                                 self.death_rate, self.reproduction_rate, self.weights, self.learning_rate,
                                 self.discount_factor, self.hunger_minimum, self.tree_function)  # ID is changed in Grid.update()
        return offspring