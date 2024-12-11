from timm import create_model
import torch.nn as nn

def get_model(config, num_classes):
    model =  create_model(
        model_name=config['model']['architecture'],
        pretrained=config['model']['pretrained'],
        num_classes=num_classes
    )
    return model