from numba import jit
import numpy as np 
import cv2
from time import time
from numpy import exp, array, random, dot

def get_bw_image(impath,dim):
    im=cv2.imread(impath)
    cop=cv2.resize(im,dim,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(cop,cv2.COLOR_BGR2GRAY)
    (thresh, bw) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    bw=bw/255
    del im,gray,cop
    return bw

@jit (nopython=True,parallel=True)
def image_to_trainning_data_set(list_of_image,trainning_data_set,dim ):
    counter=0
    c=0
    for k in list_of_image:
        for i in range(dim[0]):
            for j in range(dim[1]):
                trainning_data_set[c][counter]=k[i][j]
                counter += 1
                if(counter>=dim[0]*dim[1]):
                    c=c+1
                    counter=0
    return trainning_data_set

def sigmoid(x):
    return 1/(1+exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

@jit(nopython=True,forceobj=True,parallel=True)
def train_GPU(inputs,outputs,iteration,wt1,wt2,wt3):
    for i in range(iteration):
        output_from_layer_1 = 1/(1+exp(-1*dot(inputs, wt1)))#sigmoid(dot(inputs, wt1))#
        output_from_layer_2 = 1/(1+exp(-1*dot(output_from_layer_1,wt2)))#sigmoid(dot(output_from_layer1,wt2))
        output_from_layer_3 = 1/(1+exp(-1*dot(output_from_layer_2,wt3)))#sigmoid(dot(output_from_layer_2,wt3))#
           
        layer3_error=outputs - output_from_layer_3
        layer3_delta=layer3_error*output_from_layer_3*(1-output_from_layer_3)#sigmoid_derivative(output_from_layer_3)#

        layer2_error = layer3_delta.dot(wt3.T)
        layer2_delta = layer2_error*output_from_layer_2*(1-output_from_layer_2)# sigmoid_derivative(output_from_layer_2)#

        # Calculate the error for layer 1 (By looking at the weights in layer 1,
        # we can determine by how much layer 1 contributed to the error in layer 2).
        layer1_error = layer2_delta.dot(wt2.T)
        layer1_delta = layer1_error *output_from_layer_1*(1-output_from_layer_1)# sigmoid_derivative(output_from_layer_1)#

        # Calculate how much to adjust the weights by
        layer1_adjustment = inputs.T.dot(layer1_delta)
        layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
        layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)
        wt1 += layer1_adjustment
        wt2 += layer2_adjustment
        wt3 += layer3_adjustment
    return wt1,wt2,wt3
class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self,training_data_input,training_data_output,dim):
        #number of hidden neurons should be between number of output neurons and number of input neuron
        self.layer1 = NeuronLayer(100,len(training_data_input[0]))
        self.layer2 = NeuronLayer(5,100)
        self.layer3 = NeuronLayer(len(training_data_output[0]),5)
        self.inputs=training_data_input
        self.outputs=training_data_output
        self.dim=dim
        #we have here 105 hidden neuron 
    
    def train(self, number_of_training_iterations):
        self.layer1.synaptic_weights,self.layer2.synaptic_weights,self.layer3.synaptic_weights=train_GPU(self.inputs,self.outputs,number_of_training_iterations,self.layer1.synaptic_weights,self.layer2.synaptic_weights,self.layer3.synaptic_weights)
   
    def __sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def predict(self,image):
        inps=np.zeros([1,self.dim[0]*self.dim[1]])
        counter=0
        for i in range(self.dim[1]):
            for j in range(self.dim[0]):
                inps[0][counter]=image[i][j]
                counter = counter+1
        l1_out=self.__sigmoid(dot(inps,self.layer1.synaptic_weights))
        l2_out=self.__sigmoid(dot(l1_out,self.layer2.synaptic_weights))
        l3_out=self.__sigmoid(dot(l2_out,self.layer3.synaptic_weights))
        return np.around(l3_out)#l3_out

image_dimontion=(20,20)
im1=get_bw_image("data2.PNG",image_dimontion)
im2=get_bw_image("data4.PNG",image_dimontion)
im3=get_bw_image("data6.PNG",image_dimontion)
im4=get_bw_image("pent2.PNG",image_dimontion)

im5=get_bw_image("pent3.PNG",image_dimontion)
im6=get_bw_image("shos1.png",image_dimontion)
im7=get_bw_image("shos2.png",image_dimontion)
im8=get_bw_image("boots1.png",image_dimontion)
im9=get_bw_image("boots2.png",image_dimontion)
list_of_im=[im1,im2,im3,im4,im5,im6,im7,im8,im9]
trainning_data_set=np.zeros([len(list_of_im),image_dimontion[1]*image_dimontion[0]])

trainning_data_set=image_to_trainning_data_set(list_of_im,trainning_data_set,image_dimontion)
trainning_data_out=np.array([[0,0,0,1],
                             [0,0,0,1],
                             [0,0,0,1],
                             [0,0,1,0],
                             [0,0,1,0],
                             [0,1,0,0],
                             [0,1,0,0],
                             [1,0,0,0],
                             [1,0,0,0]])
names=["boots","shos","pent","t-shirt"]
nn=NeuralNetwork(trainning_data_set,trainning_data_out,image_dimontion)
t=time()
nn.train(60000)
tn=time()
image=get_bw_image("pull.PNG",image_dimontion)
prediction=nn.predict(image)
print("prediction :",prediction)
index=0
for i in range(len(prediction[0])):
    if(prediction[0][i]>0):
        index=i
        break
print("tis is a ",names[index])
print("trainning time ",tn-t)