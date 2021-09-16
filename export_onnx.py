
from models import networks
import torch
#set gpu
gpu_ids='0'
str_ids = gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])

model = networks.define_RoadSeg(num_labels=2, init_type='xavier',init_gain=0.02, gpu_ids=gpu_ids)
MODELPATH = "./checkpoints/kitti/kitti_net_RoadSeg.pth"
state_dict = torch.load(MODELPATH, map_location='cuda')
model.load_state_dict(state_dict, strict=False)
model.eval()

input_data= torch.randn(1, 3, 288, 224, device='cuda')

y = model(input_data)  # dry run

file_path= MODELPATH.replace('.pth', '.onnx')  # new filename
torch.onnx.export(model, input_data, file_path, verbose=False, opset_version=12)

print('ONNX export success, saved as %s' ,file_path)