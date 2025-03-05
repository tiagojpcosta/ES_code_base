
import numpy as np

# example network
class ActivationFunction:
    """
    A class containing common activation functions.
    """
    @staticmethod
    def relu(x):
        """
        Rectified Linear Unit (ReLU) activation function.
        
        Parameters:
        x (ndarray): Input array.
        
        Returns:
        ndarray: Output after applying ReLU function.
        """
        return np.maximum(0, x)

    @staticmethod
    def tanh(x):
        """
        Hyperbolic tangent (tanh) activation function.
        
        Parameters:
        x (ndarray): Input array.
        
        Returns:
        ndarray: Output after applying tanh function.
        """
        return np.tanh(x)

class MLP:
    """
    A simple Multi-Layer Perceptron (MLP) neural network.
    
    Attributes:
    layer_sizes (list): List containing the number of neurons in each layer.
    num_layers (int): Number of layers in the network.
    activation_func (function): Activation function used in hidden layers.
    weights (list): List of weight matrices for each layer.
    biases (list): List of bias vectors for each layer.
    net_scaling (float): Scaling factor for the final output.
    num_params (int): Total number of parameters in the network.
    """
    def __init__(self, layer_sizes, activation_func=ActivationFunction.tanh):
        """
        Initializes the MLP with given layer sizes and activation function.
        
        Parameters:
        layer_sizes (list): List of integers specifying the number of neurons per layer.
        net_scaling (float, optional): Scaling factor for the network output. Default is 0.01.
        activation_func (function, optional): Activation function to use. Default is tanh.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activation_func = activation_func
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for each layer
        self.num_params = 0
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.biases.append(np.random.randn(layer_sizes[i+1]))
            self.num_params += (layer_sizes[i] + 1) * layer_sizes[i+1]
    
    def __call__(self, x):
        """
        Forward propagation through the network.
        
        Parameters:
        x (ndarray): Input array.
        
        Returns:
        ndarray: Output of the MLP after forward propagation.
        """
        # Forward propagate through all layers
        for i in range(self.num_layers - 1):
            x = self.activation_func(np.dot(x, self.weights[i]) + self.biases[i])
        return  (np.dot(x, self.weights[i+1]) + self.biases[i+1])
    
    def get_params(self):
        """
        Retrieves all parameters (weights and biases) as a flattened array.
        
        Returns:
        ndarray: A 1D array containing all network parameters.
        """
        params = []
        for w, b in zip(self.weights, self.biases):
            params.extend(w.flatten())
            params.extend(b.flatten())
        return np.array(params)
    
    def set_params(self, params):
        """
        Sets the network parameters from a flattened array.
        
        Parameters:
        params (ndarray): A 1D array containing all network parameters.
        """
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            w_shape = (self.layer_sizes[i], self.layer_sizes[i+1])
            b_shape = (self.layer_sizes[i+1],)
            
            w_size = np.prod(w_shape)
            b_size = np.prod(b_shape)
            
            self.weights[i] = params[idx:idx+w_size].reshape(w_shape)
            idx += w_size
            self.biases[i] = params[idx:idx+b_size].reshape(b_shape)
            idx += b_size
