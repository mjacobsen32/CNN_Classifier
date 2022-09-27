import torch
import torchvision.models as models

class BaseModel:
    def __init__(self, device_str, **model_kwargs):
        self.device = torch.device(device=device_str) # Either 'Cuda' or 'CPU' (GPU or CPU usage)
        self.model = self.get_model(**model_kwargs) # Creating model: To be loaded from model dict
                                                    # or created from scratch with seed 
                                                    # via inference model or new model
                                        
    
    def get_model(self, **kwargs):
        model_name = kwargs['model_name']
        num_classes = kwargs['num_classes']
        if model_name == "AlexNet":
            nn = models.alexnet(num_classes=num_classes).to(self.device)
        elif model_name == 'GoogleNet':
            nn = models.googlenet(num_classes=num_classes, aux_logits=False, init_weights=True).to(self.device)
        elif model_name == 'ResNet18':
            nn = models.resnet18(num_classes=num_classes).to(self.device)
        elif model_name == 'VGG13':
            nn = models.vgg13(num_classes=num_classes).to(self.device)
        return(nn)