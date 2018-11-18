import torch.nn as nn

class ModelUtils():
    @staticmethod
    def freeze(model):
        ModelUtils.__set_req_grad(model, False)

    @staticmethod
    def unfreeze(model):
        ModelUtils.__set_req_grad(model, True)

    @staticmethod
    def __set_req_grad(model, req_grad=True):
        for param in model.parameters():
            param.requires_grad = req_grad
            
    @staticmethod
    def get_binary_classifier():
        binary_classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        return binary_classifier