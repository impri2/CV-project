import torch
from utils.torch_utils import revert_sync_batchnorm

from models.experimental import attempt_load


def save_TracedModel(weights, device=None, img_size=(640,640)):

    model = attempt_load(weights, map_location=device)
    print(" Convert model to Traced-model... ") 

    model = revert_sync_batchnorm(model)
    model.to('cpu')
    model.eval()

    detect_layer = model.model[-1]
    model.traced = True
    
    rand_example = torch.rand(1, 3, img_size, img_size)

    traced_script_module = torch.jit.trace(model, rand_example, strict=False)
    traced_script_module.save("traced_model.pt")
    print(" traced_script_module saved! ")

    return model.traced


    model = traced_script_module
    model.to(device)
    detect_layer.to(device)

def forward(self, x, augment=False, profile=False):
    out = self.model(x)
    out = self.detect_layer(out)
    return out

save_TracedModel("runs/train/yolov7_multi_res10/weights/best.pt")
