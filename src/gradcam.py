"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
os.chdir("../TRN")
import cv2
import numpy as np
import torch

from misc_functions import get_example_params, save_class_activation_images
from get_model_data import get_model_data


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        # for i, module in enumerate(self.model.base_model.modules()):
            # print("\033[31m {} : {}\033[0m".format(i, module))


    def save_gradient(self, grad):
        self.gradients = grad
    
    def forward_pass_on_convolutions(self, input):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """        
        data_dict = dict()
        data_dict[self.model.base_model._op_list[0][-1]] = input

        for i, op in enumerate(self.model.base_model._op_list):
            if op[1] != 'Concat' and op[1] != 'InnerProduct':
                data_dict[op[2]] = getattr(self.model.base_model, op[0])(data_dict[op[-1]])
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self.model.base_model, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].size())
                    raise
            if i == self.target_layer:
                data_dict[op[2]].register_hook(self.save_gradient)
                conv_output = data_dict[op[2]]
        return conv_output, data_dict[self.model.base_model._op_list[-1][2]]

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # Forward pass on the classifier
        x = self.model.new_fc(x)
        x = x.view((-1, self.model.num_segments) + x.size()[1:])
        x = self.model.consensus(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()
        print("the shape of grad is : {}".format(guided_gradients.shape))
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(2, 3))  # Take averages for each gradient 3x96
        # Create empty numpy array for cam
        cam = np.zeros([ target.shape[0],target.shape[2],target.shape[3] ], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for f in range(weights.shape[0]):
            frame_weights = weights[f]
            frame_cam = cam[f]
            for i, w in enumerate(frame_weights):
                frame_cam += w * target[f, i, :, :]
        cam = cam.transpose([1, 2, 0])     # 为了resize将时间维转到通道维
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam

if __name__ == '__main__':
    # Get params
    pretrained_model, data_loader = get_model_data()
    pretrained_model = pretrained_model.cuda()
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=213)
    # Generate cam mask
    for i, (data, target, idx) in enumerate(data_loader):
        # print
        print("Processing {}".format(i))
        # get the input, target, idx, original image
        original_image = data.numpy().copy()
        original_image = original_image.reshape([-1, 3, original_image.shape[2], original_image.shape[3]])
        data.requires_grad = True
        data = data.cuda()
        input_var = torch.autograd.Variable(data.view(-1, 3, data.size(2), data.size(3)))
        target_class = int(target.cpu().numpy().copy()[0])
        idx = int(idx.cpu().numpy().copy()[0])  
        # run the forward pass , get cam      
        cam = grad_cam.generate_cam(input_var, target_class)
        np.save("../results/test_cam.npy",cam)
        # Save mask
        file_name_to_export = "/mnt/data/liweijie/trn_visualization/"+\
                                    "trn_gradcam/{:d}_{:d}_GradCAM".format(target_class, idx)
        save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
