from flask import Flask, request, render_template, jsonify
import numpy as np
import random
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.1

# Fitness Function: Calculate Total Distance
def calculate_distance(route, locations):
    distance = 0
    for i in range(len(route) - 1):
        distance += np.linalg.norm(np.array(locations[route[i]]) - np.array(locations[route[i + 1]]))
    distance += np.linalg.norm(np.array(locations[route[-1]]) - np.array(locations[route[0]]))  # Return to start
    return distance

# Initialize Population
def initialize_population(num_locations):
    population = [random.sample(range(num_locations), num_locations) for _ in range(POPULATION_SIZE)]
    return population

# Crossover
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    pointer = 0
    for city in parent2:
        if city not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = city
    return child

# Mutation
def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Genetic Algorithm
def genetic_algorithm(locations):
    num_locations = len(locations)
    population = initialize_population(num_locations)
    for generation in range(GENERATIONS):
        # Evaluate Fitness
        fitness = [1 / calculate_distance(route, locations) for route in population]
        fitness_probs = [f / sum(fitness) for f in fitness]
        
        # Selection
        parents = [population[np.random.choice(range(POPULATION_SIZE), p=fitness_probs)] for _ in range(POPULATION_SIZE)]
        
        # Crossover and Mutation
        next_generation = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            next_generation.extend([child1, child2])
        
        population = next_generation
    
    # Return the best route
    best_route = min(population, key=lambda route: calculate_distance(route, locations))
    return best_route, calculate_distance(best_route, locations)

# Plot Route
def plot_route(locations, route, filename):
    plt.figure(figsize=(8, 6))
    route_locations = [locations[i] for i in route] + [locations[route[0]]]
    x, y = zip(*route_locations)
    plt.plot(x, y, marker='o', color='b')
    plt.scatter(x, y, color='red')
    plt.title("Optimized Delivery Route")
    plt.savefig(filename)
    plt.close()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        # Get locations from user input
        locations = request.json.get('locations')
        if not locations or len(locations) < 2:
            return jsonify({'error': 'Please provide at least 2 locations.'}), 400
        
        # Convert to float tuples
        locations = [tuple(map(float, loc)) for loc in locations]
        
        # Run Genetic Algorithm
        best_route, best_distance = genetic_algorithm(locations)
        
        # Plot the optimized route
        filename = 'static/optimized_route.png'
        plot_route(locations, best_route, filename)
        
        return jsonify({'route': best_route, 'distance': best_distance, 'image': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=9090)
