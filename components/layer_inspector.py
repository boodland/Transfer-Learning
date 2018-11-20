from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class LayerInterceptor():
    def intercept_outputs(self, layer):
        self.forward_outputs = np.array([])
        self.forward_handler = None
        self.backward_outputs = np.array([])
        self.backward_handler = None
        
        def __forward_interceptor(module, forw_input, forw_output):
            self.forward_outputs = forw_output[0].clone().detach().cpu().numpy()
            
        def __backward_interceptor(module, grad_input, grad_output):
            self.backward_outputs = grad_output[0].clone().detach().cpu().numpy()
        
        self.forward_handler = layer.register_forward_hook(__forward_interceptor)
        self.backward_handler = layer.register_backward_hook(__backward_interceptor)

class LayerInspector():
    resize_t = Resize([224, 224])
    to_tensor_t = ToTensor()
    normalizer_t = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    compose_t = Compose([resize_t, to_tensor_t, normalizer_t])
    
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.interceptors = []
        self.__register_layers()
        
    def __register_layers(self):
        for layer in self.layers:
            interceptor = LayerInterceptor()
            interceptor.intercept_outputs(layer)
            self.interceptors.append(interceptor)
            
    def __get_image_batch(self, img_path):
        img = Image.open(img_path)
        batch = self.compose_t(img)[None, :, :, :]
        return batch
    
    def predict(self, img_path, class_index):
        index2label = [class_index[str(k)][1] for k in range(len(class_index))]
        batch = self.__get_image_batch(img_path)
    
        self.model.eval()
        preds = self.model(batch)

        preds_values = preds.data.numpy()[0]
        sorted_preds = preds_values.argsort()[::-1]

        for index in sorted_preds[:5]:
            print(index, index2label[index], preds_values[index])
            
        predicted_class = preds[:, sorted_preds[0]]
        predicted_class.backward()
    