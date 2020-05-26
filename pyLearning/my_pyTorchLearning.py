import torch
#input
X = torch.Tensor([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
#output
Y = torch.Tensor([[1],[1],[0]])

#activate Function
def sigmoid(x):
    return 1/(1+torch.exp(-x))

#Derivatite(导数)
def derivatives_sigmoid(x):
    return x*(1-x)
#Variable intialization
epoch = 50000 #traning rounds
lr = 0.1 # learning rate
inputlayer_neurons = X.shape[1] #3个
hiddenlayer_neurons = 3
output_neurons = 1

#weigh and bias intialization
wh = torch.rand(inputlayer_neurons,hiddenlayer_neurons).type(torch.FloatTensor)
bh=torch.randn(1,hiddenlayer_neurons).type(torch.FloatTensor)
wout = torch.rand(hiddenlayer_neurons,output_neurons)
bout = torch.randn(1,output_neurons)

#training
for i in range(epoch):

    #Forward Propogation 前向传播
    hidden_layer_input1 = torch.mm(X,wh)
    hidden_layer_input = hidden_layer_input1+ bh # y = wX +b
    hidden_layer_activations = sigmoid(hidden_layer_input) #
    
    output_layer_input1 = torch.mm(hidden_layer_activations,wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)

    #Backpropagation
    E = Y - output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hidden_layer_activations)
    d_output  = E * slope_output_layer
    Error_at_hidden_layer = torch.mm(d_output,wout.t())
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += torch.mm(hidden_layer_activations.t(),d_output) *lr
    bout += d_output.sum()*lr
    wh += torch.mm(X.t(),d_hiddenlayer)*lr
    bh += d_output.sum()*lr

# model = torch.nn.Sequential(
#     torch.nn.Linear(inputlayer_neurons, hiddenlayer_neurons),
#     torch.nn.ReLU(),
#     torch.nn.Linear(hiddenlayer_neurons,output_neurons),
# )
loss_fn = torch.nn.CrossEntropyLoss()
print('actual:\n',Y)
print('predicted:\n',output)