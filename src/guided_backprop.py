"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
os.chdir("../TRN")
import torch
from torch.nn import ReLU
import numpy as np

from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)

from get_model_data import get_model_data
import time

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Register hook to the first layer
        for i, (pos, module) in enumerate(self.model.module.base_model._modules.items()):
            if i == 0 :
                first_layer = module
                print(first_layer)
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.model.module.base_model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)
        for pos, module in self.model.module.consensus._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)
        

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()
        return gradients_as_arr


if __name__ == '__main__':
    pretrained_model, data_loader = get_model_data()
    pretrained_model = torch.nn.DataParallel(pretrained_model.cuda())
    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    for i, (data, target, idx) in enumerate(data_loader):
        # print
        print("Processing {}".format(i))
        # get the input, target, idx
        data.requires_grad = True
        input_var = torch.autograd.Variable(data.view(-1, 3, data.size(2), data.size(3)))
        input_var = torch.autograd.Variable(data)
        target_class = int(target.cpu().numpy().copy()[0])
        idx = int(idx.cpu().numpy().copy()[0])
        start = time.time()  # 计时
        guided_grads = GBP.generate_gradients(input_var, target_class)
        end = time.time() # 计时
        print("costing time {} s".format(end - start))
        np.save("../results/test.npy",guided_grads)
        guided_grads = np.load("../results/test.npy")
        file_name_to_export = "/mnt/data/liweijie/trn_visualization/"+\
                                "trn_guidedbackpro1/{:d}_{:d}_test".format(target_class, idx)
        # Save colored gradients
        save_gradient_images(data, guided_grads, file_name_to_export + '_Guided_BP_color')
    print('Guided backprop completed')
