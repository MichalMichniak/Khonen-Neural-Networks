import numpy as np
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
def sigmoid(x : np.array) -> np.array:
    t = (1/(1+np.exp(-x)))
    return t 


def d_sigmoid(x : np.array)  -> np.array:
    sig = sigmoid(x)
    return sig*(1-sig)
    pass

class Net: 
    
    def __init__(self,num_of_neurons_per_layer : np.array, weight : List[np.array] = None, biases : List[np.array] = None, activation_func : callable = sigmoid, d_activation_func : callable = d_sigmoid, generation_dystribution_const : float = 2.0) -> None:
        """
        Creating a net

        Parameters:
            num_of_neurons_per_layer (np.array[int]): array (numpy.array) of number of neurons in each layer
            weight (List[np.array[np.array[float]]]): weights of connections between neurons weight[num of layer][num of input neuron][num of output neuron] (optional)
            biases (List[np.array[float]]): biases of each neuron in each layer biases[num of layer][num of neuron] (optional)
            activation_func (callable[[np.array[float]],np.array[float]]): activation function (default sigmoid)
            d_activation_func (callable[[np.array[float]],np.array[float]]): a derivative of the activation function (default d_sigmoid)
            generation_dystribution_const (float): constance used to generate random weights and biases (optional) 

        Returns:
            None
        """
        ## TODO: checking if weights and biases are correct
        if weight != None:
            self.weight : List[np.array] = weight
        else:
            self.weight : List[np.array] = [np.array([np.array([np.random.uniform(-generation_dystribution_const,generation_dystribution_const) for k in range(num_of_neurons_per_layer[i+1])],dtype=float) for j in range(num_of_neurons_per_layer[i])]) for i in range(len(num_of_neurons_per_layer) - 1)]

        if biases != None:
            self.biases : List[np.array] = biases
        else:
            self.biases : List[np.array] = [np.array([np.random.uniform(-generation_dystribution_const,generation_dystribution_const) for j in range(i)],dtype=float) for i in num_of_neurons_per_layer]
            pass
        
        self.activation_func : callable[[np.array],np.array] = activation_func
        self.d_activation_func : callable[[np.array],np.array] = d_activation_func
        self.neurons_sum : List[np.array] = [np.array([0.0 for j in range(i)]) for i in num_of_neurons_per_layer]
        self.neurons_out : List[np.array] = [np.array([0.0 for j in range(i)]) for i in num_of_neurons_per_layer]
        self.neurons_error : List[np.array] = [np.array([0.0 for j in range(i)]) for i in num_of_neurons_per_layer]
        self.num_neur_layer : np.array = num_of_neurons_per_layer
        self.n = 0.05
        pass

    def iter(self,x_param : np.array, y_param : np.array,*args) -> None:
        """
        One iteration of training set

        Parameters:
            x_param (np.array[float]): input parameters
            y_param (np.array[float]): output parameters

        Returns:
            None
        """
        self.back_propagation(self.generate_output(x_param),y_param)

    def back_propagation(self, result_param : np.array, y_param : np.array) -> None:
        """
        Back propagation algorithm

        Parameters:
            y_param (np.array[float]): output parameters
            result_param (np.array[float]): predicted output parameters
        
        Returns:
            None
        """
        self.neurons_error[-1] = (result_param - y_param)*self.d_activation_func(self.neurons_sum[-1])
        if len(self.num_neur_layer)-2>0:
            for i in reversed(range(len(self.num_neur_layer)-2)):
                self.neurons_error[i+1] =  self.d_activation_func(self.neurons_sum[i+1])*(self.weight[i+1]@self.neurons_error[i+2])
        for i in range(len(self.weight)):
            dw =self.n*np.reshape(self.neurons_sum[i], (self.neurons_sum[i].shape[0], 1))*self.neurons_error[i+1].T
            self.weight[i] = self.weight[i] - dw
        
        for i in range(len(self.biases)-1):
            self.biases[i+1] -= self.neurons_error[i+1]*self.n

    def generate_output(self, x_param : np.array) -> np.array:
        """
        Get net prediction on given parameters

        Parameters:
            x_param (np.array[float]): input parameters
        
        Returns:
            np.array[float]: net prediction
        """
        self.neurons_sum[0] = x_param
        self.neurons_out[0] = self.activation_func(self.neurons_sum[0])
        for i in range(len(self.num_neur_layer)-1):
            self.neurons_sum[i+1] = self.neurons_out[i]@self.weight[i]+self.biases[i+1]
            self.neurons_out[i+1] = self.activation_func(self.neurons_sum[i+1])
        return self.neurons_out[-1]

