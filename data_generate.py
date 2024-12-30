import pickle
import numpy as np
from config import get_args


args = get_args()
n_instances = args.n_instances #the number of circuits
n_nodes = args.n_nodes #the number of test nodes of each instance 
n_faults = args.n_faults #the number of faults in each instance
cost_max = args.cost_max #the maximum cost of nodes in the circuit
cost_range = args.cost_range #the range of cost of nodes in the circuit

acc_thr = args.acc_thr #the accuracy threshold
acc_generated = np.random.rand(int(args.n_instances), int(args.n_nodes), int(args.n_faults)) # generate the accuracy of each node
cost_generated = np.random.randint(low = args.cost_range[0], high = args.cost_range[1], size =(int(args.n_instances), int(args.n_nodes))) # generate the cost of each node

data = {
    'n_instances': n_instances,
    'acc_generated': acc_generated,
    'cost_generated': cost_generated,
    'cost_max': cost_max
}

with open('instances%d_nodes%d_cost_max%d_acc_thr%.1f_data.pkl'%(int(n_instances), int(n_nodes), int(cost_max), acc_thr), 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
print('Data generated and saved to instances%d_nodes%d_cost_max%d_acc_thr%.1f_data.pkl'%(int(n_instances), int(n_nodes), int(cost_max), acc_thr))
