import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Generate data for the circuit nodes selection problem')
    parser.add_argument('--n_instances', type=int, default=100, help='the number of instances')
    parser.add_argument('--n_nodes', type=int, default=10, help='the number of nodes in each instance')
    parser.add_argument('--cost_max', type=int, default=100, help='the maximum total cost of nodes in the circuit')
    parser.add_argument('--cost_range', type=int, default=(1, 100), help='the range of cost of nodes in the circuit')
    parser.add_argument('--acc_thr', type=float, default=0.5, help='the accuracy threshold')
    parser.add_argument('--n_faults', type=int, default=13, help='the number of faults in each instance')
    parser.add_argument('--alpha', type=float, default=0.5, help='the hyperparameter of cost')
    parser.add_argument('--beta', type=float, default=0.5, help='the hyperparameter of accuracy')
    parser.add_argument('--episodes', type=int, default=1000, help='the number of episodes')
    args = parser.parse_args()
    return args

