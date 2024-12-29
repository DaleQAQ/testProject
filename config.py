import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Generate data for the circuit nodes selection problem')
    parser.add_argument('--n_instances', type=int, default=100)
    parser.add_argument('--n_nodes', type=int, default=10)
    parser.add_argument('--cost_max', type=int, default=100)
    parser.add_argument('--cost_range', type=int, default=(1, 100))
    parser.add_argument('--acc_thr', type=float, default=0.5)
    args = parser.parse_args()
    return args

