from ortools.sat.python import cp_model
from scipy.spatial import distance
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import routing_parameters_pb2
from ortools.constraint_solver import pywrapcp
import sys
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import dirname, realpath


class Solution:

    def __init__(self, cost_to_build, tot_map):
        self.costToBuild = cost_to_build  # optimal cost to build in solution
        self.totMap = tot_map  # 0101010101010101010110
        self.mapOfChosen = np.where(tot_map == 1)[0]
        self.optNumTruck = 0  # optimal value is the right division

        self.minCostTrucks = 0
        self.minNumTruck = 0  # number of truck in solution min
        self.optKmTruck = 0  # number of km in solution in
        self.planRoute = []  # plan route of min problem
        self.totalCost = 0

    def get_plan_route(self):
        return self.planRoute

    def get_map_of_chosen(self):
        return self.mapOfChosen

    def set_plan_route(self, value):
        self.planRoute = value

    def set_min_cost_trucks(self, value):
        self.minCostTrucks = value
        self.totalCost = self.minCostTrucks + self.costToBuild

    def set_opt_num_truck(self, value):
        self.optNumTruck = value
        self.minNumTruck = optNumTruck

    def set_opt_km_truck(self, value):
        self.optKmTruck = value

    def set_new_min_route(self, km, n_truck, planned_route, cost):
        self.optKmTruck = km
        self.minNumTruck = n_truck
        self.planRoute = planned_route
        self.minCostTrucks = cost

    def show(self):
        toWrite_formatted = ""
        toWrite_auto = ""

        toWrite_formatted += "Solution with total cost: " + str(self.totalCost) + "\n"
        toWrite_auto += str(self.totalCost) + "\n"

        toWrite_formatted += "Cost to open markets: " + str(self.costToBuild) + "\n"
        toWrite_auto += str(self.costToBuild) + "\n"

        toWrite_formatted += "Cost to refurbish: " + str(self.minCostTrucks) + "\n"
        toWrite_auto += str(self.minCostTrucks) + "\n"

        toWrite_formatted += "Chosen markets: "

        for i in range(len(self.mapOfChosen)):
            toWrite_formatted += str(self.mapOfChosen[i] + 1)
            toWrite_auto += str(self.mapOfChosen[i] + 1)
            if i != len(self.mapOfChosen)-1:
                toWrite_formatted += ","
                toWrite_auto += ","
            else:
                toWrite_formatted += "\n"
                toWrite_auto += "\n"

        for i in range(len(self.planRoute)):
            toWrite_formatted += "Route of vehicle " + str(i + 1) + ": "
            toWrite_formatted += str(self.mapOfChosen[self.planRoute[i][0]] + 1)
            toWrite_auto += str(self.mapOfChosen[self.planRoute[i][0]] + 1)
            for k in range(1, len(self.planRoute[i])):
                toWrite_formatted += "," + str(self.mapOfChosen[self.planRoute[i][k]] + 1)
                toWrite_auto += "," + str(self.mapOfChosen[self.planRoute[i][k]] + 1)
            toWrite_formatted += "\n"
            toWrite_auto += "\n"

        return toWrite_formatted, toWrite_auto


if __name__ == '__main__':

    filepath = realpath(__file__)

    dir_of_file = dirname(filepath)
    parent_dir_of_file = dirname(dir_of_file)
    parents_parent_dir_of_file = dirname(parent_dir_of_file)

    input_path = os.path.join(parents_parent_dir_of_file, 'input.dat')

    with open(input_path) as file:
        f = file.read()

    fl = f.split('\n')

    params = {}

    for l in fl[:5]:
        l_r = l.replace('\t', '') \
            .replace('param', '') \
            .replace(';', '') \
            .replace(':', '') \
            .replace(' ', '')
        param = l_r.split('=')

        # noinspection PyBroadException
        try:
            params[param[0]] = int(param[1])
        except:
            print("Error in file parsing!")
            sys.exit()

    cities = list()

    for l in fl[7:-1]:
        city = l.split('\t')[1:-1]

        # noinspection PyBroadException
        try:
            city = [int(i) for i in city]
            cities.append(city)
        except:
            print("Error in file parsing!")
            sys.exit()


    def dist(c1, c2) -> float:
        c1 = c1[:2]
        c2 = c2[:2]
        d = distance.euclidean(c1, c2)
        return round(d, 2)

    distances = np.zeros((len(cities), len(cities)))
    for i in range(len(cities)):
        for j in range(i, len(cities)):
            distances[i][j] = dist(cities[i], cities[j])
    distances = distances + distances.transpose()

    # Creates the model that minimize the cost of building
    model_cost = cp_model.CpModel()

    # Creates the variables.
    x_1 = {}
    for i in range(params['n']):
        x_1[i] = model_cost.NewIntVar(0, 1, 'x%i' % i)

    # Constraints
    for i in range(1, params['n']):
        if cities[i][3] == 0:
            model_cost.Add(x_1[i] == 0)

    model_cost.Add(
        x_1[0] == 1
    )

    for i in range(params['n']):
        gen = (j for j in range(params['n']) if (distances[i][j] < params['range']))
        model_cost.Add(
            sum(x_1[j] for j in gen) >= 1
        )

    model_cost.Minimize(sum(cities[i][2] * x_1[i] for i in range(params['n'])))

    # Creates the model that minimize the number of stores
    model_num = cp_model.CpModel()

    # Creates the variables.
    x_2 = {}
    for i in range(params['n']):
        x_2[i] = model_num.NewIntVar(0, 1, 'x%i' % i)

    # Constraints
    for i in range(1, params['n']):
        if cities[i][3] == 0:
            model_num.Add(x_2[i] == 0)

    model_num.Add(
        x_2[0] == 1
    )

    for i in range(params['n']):
        gen = (j for j in range(params['n']) if (distances[i][j] < params['range']))
        model_num.Add(
            sum(x_2[j] for j in gen) >= 1
        )

    model_num.Minimize(sum(x_2[i] for i in range(params['n'])))


    def find_solutions_cost(model):
        # Creates a solver and solves the model.
        solver = cp_model.CpSolver()

        # new obj
        solutions_cost = []

        for _ in tqdm(range(40)):
            status = solver.Solve(model)
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                sol_cost = np.empty(params['n'])
                for i in range(params['n']):
                    sol_cost[i] = solver.Value(x_1[i])

                solutions_cost.append(Solution(solver.ObjectiveValue(), sol_cost))

                gen_cost = (p for p in range(params['n']) if sol_cost[p] != 0)
                gen1 = (k for k in range(params['n']) if sol_cost[k] == 0)

                model.Add(
                    sum((int(sol_cost[p]) - x_1[p]) for p in gen_cost) + sum(
                        (x_1[k] - int(sol_cost[k])) for k in gen1) != 0
                )

            else:
                print('No solution found.')
                break
        return solutions_cost


    def find_solutions_num(model):
        # Creates a solver and solves the model.
        solver = cp_model.CpSolver()

        # new obj
        solutions_num = []

        for _ in tqdm(range(40)):
            status = solver.Solve(model)
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                sol_num = np.empty(params['n'])
                for i in range(params['n']):
                    sol_num[i] = solver.Value(x_2[i])

                total = 0
                for i in range(len(sol_num)):
                    if sol_num[i] == 1:
                        total += cities[i][2]
                solutions_num.append(Solution(total, sol_num))

                gen_num = (p for p in range(params['n']) if sol_num[p] != 0)
                gen1 = (k for k in range(params['n']) if sol_num[k] == 0)

                model.Add(
                    sum((int(sol_num[p]) - x_2[p]) for p in gen_num)
                    + sum((x_2[k] - int(sol_num[k])) for k in gen1) != 0
                )

            else:
                print('No solution found.')
                break
        return solutions_num


    mySolutions_cost = find_solutions_cost(model_cost)
    mySolutions_num = find_solutions_num(model_num)
    mySolutions = mySolutions_cost + mySolutions_num

    def array_equal(array1, array2):
        div = True
        for i in range(len(array1)):
            if array1[i] != array2[i]:
                div = False
                break
        return div


    mySolutions = np.array(mySolutions)
    indx = []

    for i in range(0, 40):
        for j in range(40, len(mySolutions)):
            choose1 = np.array(mySolutions[i].mapOfChosen)
            choose2 = np.array(mySolutions[j].mapOfChosen)
            if choose2.shape == choose1.shape:
                if array_equal(choose1, choose2):
                    indx.append(j)
    mySolutions = np.delete(mySolutions, indx)

    def print_solution(num_vehicle, manager, routing, solution):
        max_route_distance = 0
        total_distance = 0
        plans = []

        for vehicle_id in range(num_vehicle):
            n = 0
            index = routing.Start(vehicle_id)
            plan_output = []
            route_distance = 0.
            while not routing.IsEnd(index):
                n += 1
                plan_output.append(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            if n == 1:
                return 0, 0, True

            plan_output.append(manager.IndexToNode(index))
            plans.append(plan_output)
            max_route_distance = max(route_distance, max_route_distance)
            total_distance += route_distance
        return total_distance / 100, plans, False


    def route(distance_matrix, num_vehicle, demand, vehicle_capacity):
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix),
                                               num_vehicle, 0)

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index):
            return demand

        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)

        # Add Distance constraint.
        dimension_name = 'Capacity'
        routing.AddDimension(
            demand_callback_index,
            0,  # no slack
            vehicle_capacity,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(2)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            return print_solution(num_vehicle, manager, routing, solution)
        else:
            print('No solution found !')

    real_capacity = params['capacity']
    params.update({'capacity': real_capacity + 1})
    for sol in tqdm(mySolutions):
        length = len(sol.mapOfChosen)
        dist_matrix = np.zeros((length, length))
        row = -1
        col = -1
        for chosen1 in sol.mapOfChosen:
            row += 1
            col = -1
            for chosen2 in sol.mapOfChosen:
                col += 1
                dist_matrix[row][col] = distances[chosen1][chosen2] * 100

        optNumTruck = math.ceil((length - 1) / real_capacity)
        sol.set_opt_num_truck(optNumTruck)

        total_dist = 0
        plan_route = []

        try:
            total_dist, plan_route, _ = route(dist_matrix, optNumTruck, 1, params['capacity'])
        except:
            exit(0)

        actual_cost = sol.optNumTruck * params['Fc'] + total_dist * params['Vc']
        sol.set_new_min_route(total_dist, sol.optNumTruck, plan_route, actual_cost)
        sol.set_min_cost_trucks(actual_cost)

        for v in range(sol.optNumTruck + 1, length):
            total_dist, plan_route, self_loop = route(dist_matrix, v, 1, params['capacity'])
            if not self_loop:
                actual_cost = v * params['Fc'] + total_dist * params['Vc']
                if actual_cost < sol.minCostTrucks:
                    sol.set_new_min_route(total_dist, v, plan_route, actual_cost)
                    sol.set_min_cost_trucks(actual_cost)
            else:
                break

    best_sol = min(mySolutions, key=lambda item: item.totalCost)

    str_formatted, str_auto = best_sol.show()

    rs = best_sol.get_plan_route()
    mc = best_sol.get_map_of_chosen()
    nm = []

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for i in range(len(cities)):
        if i not in mc:
            nm.append(cities[i])

    x_points = []
    y_points = []
    for i in nm:
        x_points.append(i[0])
        y_points.append(i[1])

    ax1.scatter(x_points, y_points)

    x_points = []
    y_points = []
    for i in mc:
        x_points.append(cities[i][0])
        y_points.append(cities[i][1])

    ax1.scatter(x_points[0], y_points[0], c="red")
    ax1.scatter(x_points[1:], y_points[1:], c="violet")

    color = ['r', 'g', 'b', 'c', 'm', 'y']
    i = 0
    for r in rs:
        x_r = []
        y_r = []
        col = color[i % 6]
        for c in r:
            x_r.append(cities[mc[c]][0])
            y_r.append(cities[mc[c]][1])
        ax1.plot(x_r, y_r, c=col)
        i += 1

    img_path = os.path.join(parents_parent_dir_of_file, 'route.png')

    plt.savefig(img_path)

    form_path = os.path.join(parents_parent_dir_of_file, 'output_formatted.txt')

    with open(form_path, 'w') as f_out_formatted:
        f_out_formatted.write(str_formatted)

    out_path = os.path.join(parents_parent_dir_of_file, 'output.txt')

    with open(out_path, 'w') as f_out:
        f_out.write(str_auto)
