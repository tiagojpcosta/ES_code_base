import numpy as np

def evaluate_individual(model_params, model, target, initial_condition,  N_internal):
    model.set_params(model_params)
    fitness = 0
    for i in range(N_internal):
        fitness_sim = fitness_evaluator(model, target[i], initial_condition[i])
        fitness += fitness_sim 
    return fitness/N_internal



def fitness_evaluator(model, target, initial_position):
    # Here you would write your function with everything you need
    target_f = target[0]*initial_position**4 + target[1]*initial_position**3 + target[2]*initial_position**2 + target[3]*initial_position**1 + target[4]

    loss = (target_f-model(initial_position.reshape(len(initial_position),1))[:,0])**2
    return loss.mean()


def sample_initial_position(N_internal, params):
    #this are example functions
    # in this case the initial position is just random points in a range defined by the parameters
    max_x = params['max_x']
    min_x = params['min_x']
    x = np.random.rand(N_internal, 30)*(max_x-min_x) + min_x
    return x

def sample_target(N_internal, params= None):
    #this are example functions
    # in this case the target are some fixed coeficients of a quadratic function
    # The params here are not being used but i left it for completion 
    target = np.tile(np.array([20,-5,1,3,5]), (N_internal, 1))
    return target
