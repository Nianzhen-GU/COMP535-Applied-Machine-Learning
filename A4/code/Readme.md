#Package used 

There are two different jupyter notebook files in the zip folder, GON, which is the original code listed in the authors' github repository, and GON_Validate, which is the file created by us on top of the original code in order to validate the result. 

To experiment with recreating the results, we mainly used the GON model. We conducted the same experiment on Variational GON and Implicit GON but the result varies a lot from the results in paper. 

The main model is built by torch, and the dataset used to test was MNIST and Fashion MNIST. The author of the paper never mentioned anything about the hyperparameter about the results, we got the similar results by tuning the hyperparameters. 

#How to get the same results

We tuned the number of the latent variables multiple times. 

To get the results of comparing different models, we manually changed the activation functions in the neural networks with different latent variables to get the graph in report. We run it on MNIST training dataset. 