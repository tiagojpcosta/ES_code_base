import numpy as np
import multiprocessing as mp
import timeit
from scipy.linalg import eigh
from utils_ES import sample_initial_position, sample_target, fitness_evaluator, evaluate_individual

def cmaes_parallel(model, N_internal, max_it, fitting_params, sim_params, env=None, print_fitness=False):
    """
    Perform parameter optimization using a parallelized CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.
    
    This function implements a parallelized version of CMA-ES to optimize the parameters of a given model.
    Each candidate solution (individual) is evaluated using multiple initial conditions and targets, where N_internal
    specifies how many evaluations are performed per individual. The algorithm continues until the maximum number 
    of evaluations is reached, a desired fitness threshold is achieved, or the maximum number of iterations is met.
    
    Parameters:
    -----------
    model : object
        The model to be optimized. It must have the attributes:
            - num_params: an integer representing the number of parameters.
            - get_params(): returns the current parameter vector.
            - set_params(params): updates the model with the new parameters.
    N_internal : int
        Number of internal evaluations (i.e., number of initial conditions and target pairs) used to compute 
        the fitness of each candidate.
    max_it : int
        Maximum number of iterations (generations) for the algorithm.
    fitting_params : dict
        Dictionary containing optimization parameters:
            - 'sigma': Standard deviation used for generating candidate solutions.
            - 'stopfitness': Fitness threshold to stop the optimization early.
            - 'saving_every_n_epochs': Frequency (in iterations) to save intermediate parameters and loss.
            - (Other keys may be added as needed.)
    sim_params : dict
        Simulation parameters for generating initial conditions and targets using the functions:
            - sample_initial_position()
            - sample_target()
    env : optional
        This would be your mujoco  env.
    print_fitness : bool, optional
        If True, prints iteration details including mean fitness and iteration duration.
    
    Returns:
    --------
    model : object
        The optimized model with its parameters updated to the best found solution.
    
    Notes:
    ------
    - The algorithm uses multiprocessing to evaluate candidates in parallel.
    - The covariance matrix is adapted over iterations to shape the sampling distribution.
    - Intermediate results (loss and parameters) are saved periodically based on 'saving_every_n_epochs'.
    """

    # Initialize loss history over iterations
    loss_history = np.zeros(max_it + 1)
    
    # Get the number of parameters and the current mean parameter vector from the model.
    N = model.num_params
    xmean = model.get_params()

    # Retrieve optimization parameters from the fitting_params dictionary.
    sigma = fitting_params['sigma']
    stopfitness = fitting_params['stopfitness']
    stopeval = 1e3 * N ** 2  # Maximum number of function evaluations (heuristic)

    # Set the population size (lambda) using a heuristic based on the number of parameters.
    lambda_ = 4 + int(3 * np.log(N))
    print(lambda_)
    # Print information about parallel evaluation: number of external simulations and available CPUs.
    print('num external sims:', lambda_, ', how many cpus available:', mp.cpu_count(), 
          ', CPUs used:', min(mp.cpu_count()-2, lambda_), 's')

    # Determine the number of top-performing individuals to use for updating (mu) and compute their weights.
    mu = lambda_ // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    mu = int(np.floor(mu))
    weights /= np.sum(weights)  # Normalize weights
    mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)  # Effective number of parents

    # Strategy parameter settings: adaptation constants for evolution paths and covariance matrix.
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)     # Time constant for cumulation for the covariance matrix
    cs = (mueff + 2) / (N + mueff + 5)                   # Time constant for cumulation for step-size control
    c1 = 2 / ((N + 1.3) ** 2 + mueff)                    # Learning rate for rank-one update of covariance matrix
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))  # Learning rate for rank-mu update
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs  # Damping for step-size update

    # Initialize evolution paths for covariance matrix (pc) and step-size (ps)
    pc = np.zeros(N)
    ps = np.zeros(N)
    # Initialize covariance matrix factors: B (eigenbasis) and D (diagonal standard deviations)
    B = np.eye(N)
    D = np.ones(N)
    # Covariance matrix and its inverse square root.
    C = B @ np.diag(D ** 2) @ B.T
    invsqrtC = B @ np.diag(D ** -1) @ B.T
    eigeneval = 0  # Counter for tracking when to perform eigen-decomposition
    chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))  # Expectation of ||N(0,I)||

    counteval = 0  # Total number of function evaluations
    # Set up multiprocessing pool using available CPUs (leaving two cores free).
    pool = mp.Pool(min(mp.cpu_count() - 2, lambda_))
    it = 0  # Iteration counter

    # Main loop: iterate until maximum evaluations, desired fitness, or maximum iterations reached.
    while counteval < stopeval:
        starttime = timeit.default_timer()
        
        # Generate candidate solutions: for each individual in the population,
        # perturb the current mean using a scaled sample from the multivariate normal distribution.
        arx = np.array([xmean + sigma * B @ (D * np.random.randn(N)) for k in range(lambda_)])

        # Generate a common set of initial conditions and targets for evaluation.
        initial_positions = sample_initial_position(N_internal, sim_params)
        targets = sample_target(N_internal)
        
        # Build the list of inputs for parallel evaluation.
        # Each candidate (with its specific perturbed parameters) is paired with the same targets and initial positions.
        input_list = [(arx[k], model, targets, initial_positions, N_internal) for k in range(lambda_)]
        
        # Evaluate each candidate's fitness in parallel.
        arfitness = np.array(pool.starmap(evaluate_individual, input_list))

        # Optionally print the iteration number, mean fitness, and duration.
        if print_fitness:
            print('iteration number:', it, ', mean fitness:', arfitness.mean(), 
                  ', duration:', round(timeit.default_timer() - starttime, 2))
        loss_history[it] = arfitness.mean()  # Record mean fitness (loss) for this iteration

        # Update the total number of function evaluations.
        counteval += lambda_

        # Sort candidate solutions based on their fitness (lower loss is better).
        arindex = np.argsort(arfitness)
        xold = xmean.copy()  # Store previous mean for evolution path update
        
        # Recombine the top mu individuals (weighted average) to form the new mean.
        xmean = arx[arindex[:mu], :].T @ weights

        # Update the evolution path for step-size (ps)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xold) / sigma
        
        # Check for a successful evolution step 
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lambda_)) / chiN < 1.4 + 2 / (N + 1)
        
        # Update the evolution path for covariance matrix (pc)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma

        # Compute the weighted difference of the top mu individuals for covariance matrix update.
        artmp = (1 / sigma) * (arx[arindex[:mu], :].T - xold[:, np.newaxis])
        
        # Update the covariance matrix using both rank-one and rank-mu updates.
        C = (1 - c1 - cmu) * C + \
            c1 * (pc[:, np.newaxis] @ pc[:, np.newaxis].T + (1 - hsig) * cc * (2 - cc) * C) + \
            cmu * artmp @ np.diag(weights) @ artmp.T

        # Update the step-size sigma using an exponential update rule.
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        # Perform eigen-decomposition of the covariance matrix periodically to update B and D.
        if counteval - eigeneval > lambda_ / (c1 + cmu) / N / 10:
            eigeneval = counteval
            # Ensure symmetry of C before eigen-decomposition.
            C = np.triu(C) + np.triu(C, 1).T
            D, B = eigh(C)
            D = np.sqrt(D)
            invsqrtC = B @ np.diag(D ** -1) @ B.T

        # Check stopping conditions: fitness threshold, covariance matrix degeneration, or iteration limit.
        if arfitness[0] <= stopfitness or np.max(D) > 1e7 * np.min(D) or it == max_it:
            print(arfitness[0], stopfitness, np.max(D), 1e7 * np.min(D), it, max_it)
            break

        # Save intermediate results if the current iteration matches the saving frequency.
        if (it % fitting_params['saving_every_n_epochs']) == fitting_params['saving_every_n_epochs'] - 1:
            np.save('loss.npy', loss_history)
            np.save('net_params_' + str(it + 1) + '.npy', xmean)
        it += 1
 
    # Close the multiprocessing pool.
    pool.close()
    pool.join()
    
    # Set the model parameters to the best found candidate (with the lowest loss).
    xmin = arx[arindex[0], :]
    model.set_params(xmin)
    return loss_history


def antithetic_es_parallel(model, N_internal, max_it, fitting_params, sim_params, env=None, print_fitness=False):
    """
    Evolution Strategies (ES) without CMA based on the OpenAI paper 
    "Evolution Strategies as a Scalable Alternative to Reinforcement Learning" 
    using antithetic sampling.

    This function optimizes the parameters of a model using an ES approach that 
    employs bidirectional perturbations. For every noise vector sampled in one 
    direction, its negative is also sampled to improve the variance of the gradient 
    estimate and overall optimization performance.

    Parameters:
    -----------
    model : object
        The model to be optimized. It must provide:
            - get_params(): returns the current parameter vector.
            - set_params(params): updates the model with new parameters.
    N_internal : int
        Number of initial conditions/targets used to evaluate each candidate.
    max_it : int
        Maximum number of iterations.
    fitting_params : dict
        Dictionary containing optimization parameters:
            - 'sigma': Standard deviation for the perturbations.
            - 'stopfitness': A target loss value at which to stop the optimization.
            - 'learning_rate': (optional) Learning rate (default is 0.01).
            - 'lambda': (optional) Total number of perturbations (will be made even; 
            default is 4 + int(3 * log(N))).
            - 'saving_every_n_epochs': (optional) Frequency at which parameters and loss 
            are saved.
    sim_params : dict
        Dictionary of simulation parameters used by sample_initial_position().
    env : optional
        This would be your Mujoco environment (if applicable).
    print_fitness : bool, optional
        If True, prints loss and timing information for each iteration.

    Returns:
    --------
    model : object
        The model with updated parameters.
    """
    # Get the number of parameters and the current parameter vector.
    N = model.num_params
    xmean = model.get_params()

    sigma = fitting_params['sigma']
    alpha = fitting_params['learning_rate']
    stopfitness = fitting_params['stopfitness']
    stopeval = fitting_params.get('stopeval', 1e3 * N ** 2)

    # Determine population size (lambda); ensure it's an even number for antithetic sampling.
    lambda_ = fitting_params.get('lambda', 4 + int(3 * np.log(N)))
    if lambda_ % 2 == 1:
        lambda_ += 1

    loss_history = np.zeros(max_it + 1)
    counteval = 0
    it = 0

    # Create a multiprocessing pool.
    pool = mp.Pool(min(mp.cpu_count() - 2, lambda_))
    
    while counteval < stopeval:
        starttime = timeit.default_timer()
        
        # Sample half the number of noise vectors
        half = lambda_ // 2
        noise_half = np.random.randn(half, N)
        # Create antithetic (bidirectional) perturbations
        noise = np.concatenate([noise_half, -noise_half], axis=0)
        
        # Generate candidate parameter vectors: xmean + sigma * noise
        arx = np.array([xmean + sigma * noise[i] for i in range(lambda_)])
        
        # Sample the common initial conditions and targets for this iteration.
        initial_positions = sample_initial_position(N_internal, sim_params)
        targets = sample_target(N_internal)
        
        # Build the input list for parallel evaluation:
        # Each candidate (with its corresponding perturbed parameters) is evaluated.
        input_list = [(arx[i], model, targets, initial_positions, N_internal) for i in range(lambda_)]
        
        # Evaluate loss (fitness) for each candidate in parallel.
        fitness_values = np.array(pool.starmap(evaluate_individual, input_list))
        # Since lower loss is better, we define reward as the negative loss.
        rewards = -fitness_values
        
        # Compute the gradient estimate using antithetic sampling.
        # Note: With our pairing, this is equivalent to:
        #    grad = (1 / (2m * sigma)) * sum_{i=1}^{m} [ R_i^+ - R_i^- ] * epsilon_i
        # and here lambda_ = 2*m.
        grad = np.dot(rewards, noise) / (lambda_ * sigma)
        
        # Update the parameter vector (gradient ascent on reward).
        xmean = xmean + alpha * grad

        # Logging and tracking.
        mean_loss = fitness_values.mean()
        if print_fitness:
            print('Iteration:', it, 'Mean loss:', mean_loss, 
                  'Duration:', round(timeit.default_timer() - starttime, 2))
        loss_history[it] = mean_loss
        counteval += lambda_

        # Stop if the loss is below the threshold or if maximum iterations reached.
        if mean_loss <= stopfitness or it == max_it:
            print("Stopping: mean loss {} <= stopfitness {} or max iterations reached".format(mean_loss, stopfitness))
            break

        # Optionally, save intermediate results.
        if (it % fitting_params.get('saving_every_n_epochs', max_it + 1)) == (fitting_params.get('saving_every_n_epochs', max_it + 1) - 1):
            np.save('loss.npy', loss_history)
            np.save('net_params_' + str(it + 1) + '.npy', xmean)

        it += 1

    pool.close()
    pool.join()
    
    # Set the model parameters to the final updated value.
    model.set_params(xmean)
    return loss_history