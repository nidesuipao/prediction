import torch

from modeling.vectornet import HGNN
from dataset import GraphDataset
from torch_geometric.loader import DataLoader
import json

import os


model = HGNN(8, 60,3,1,64,64,64)
#state = torch.load('/home/nio/Documents/yet-another-vectornet-master/trained_params/2023428/epoch_999.valminade_1.057.pth')
#model.load_state_dict(state['state_dict'])
model.eval()

VAL_DIR = os.path.join('interm_data', 'val_intermediate')
test_data = GraphDataset(VAL_DIR)
test_loader = DataLoader(test_data, batch_size=1)
    
    
f2 = open('new_json.json', 'r')
jdata = json.load(f2)

data = dict()
data['x'] = torch.tensor(jdata['x'], dtype=torch.float32)
data['edge_index'] = torch.tensor(jdata['edge_index'], dtype=torch.int64)
data['y'] = torch.tensor(jdata['y'], dtype=torch.float32)
data['cluster'] = torch.tensor(jdata['cluster'], dtype=torch.int64)
data['valid_len'] = torch.tensor(jdata['valid_len'], dtype=torch.int64)
data['time_step_len'] = torch.tensor(jdata['time_step_len'], dtype=torch.int64)
data['batch'] = torch.tensor(jdata['batch'], dtype=torch.int64)
data['ptr'] = torch.tensor(jdata['ptr'], dtype=torch.int64)


print(model(data['x'], data['edge_index'], data['y'], data['cluster'],data['valid_len'],data['time_step_len'],data['batch'],data['ptr']))

#torch.onnx.export(model, x, 'model.onnx')
#print(data)

traced_script_module = torch.jit.trace(model, (data['x'], data['edge_index'], data['y'], data['cluster'],data['valid_len'],data['time_step_len'],data['batch'],data['ptr']))



#print(traced_script_module(data['x'], data['edge_index'], data['y'], data['cluster'],data['valid_len'],data['time_step_len'],data['batch'],data['ptr']).size())

data = next(iter(test_loader))
#y = torch.cat([data.y], 0).view(-1, 60)

traced_script_module = torch.jit.trace(model, (data.x, 
        data.edge_index,
        data.y,
        data.cluster,
        data.valid_len,
        data.time_step_len,
        data.batch,
        data.ptr))


traced_script_module.save("model.pt")




