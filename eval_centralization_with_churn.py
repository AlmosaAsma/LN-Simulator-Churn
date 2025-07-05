"""
    Procedure starting Network Simulator that adds many nodes using specified autopilot strategy and evaluates the
    effects on the centralization of the network in the presence of churn 
"""

import sys
import time
import graph_creation as gc
import networkx as nx
import network_simulator as ns
from pathlib import Path


def start_eval(strategy='random', churn_rate=0.956, tx_amt=100, interval=1000, n=10000, cpus=None, graph_id=1, seed_val=23):
    """
        Input:  'strategy' used by the autopilot to connect 'n' new nodes to the network
		'churn_rate used by the network simulator to remove nodes from the network graph at regular intervals  
            ~>  starts simulator and adds nodes one after the other; evaluates centralization of the network
                every 'interval' steps (including 1000 transactions with 'tx_amt' satoshis)
    """
    
    graph_paths = [
        './lngraph_2020_05_01__10_00.json',
    ]
    
    base_folder = f'./centralization_results/{strategy}/'
    Path(base_folder).mkdir(parents=True, exist_ok=True)
    
    for i, graph_path in enumerate(graph_paths):
        graph_folder = f'{base_folder}/graph_{i+1}/'   
        Path(graph_folder).mkdir(parents=True, exist_ok=True)

        # open file and create multigraph model of the Lightning Network:
        g = gc.parse_multi_di_graph(graph_path, exclude_disabled=True)  # EXCLUDE DISABLED!!!
        print(f"Graph {i+1} has {g.number_of_nodes()} nodes, {g.number_of_edges()} edges and {g.number_of_edges() / 2} active channels ")

        # reduce graph to largest component:
        max_comp = max(nx.strongly_connected_components(g), key=len)
        g = g.subgraph(max_comp).copy()
        print(f"Graph {i+1}: Largest connected component has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")

        start = time.perf_counter()
        ns.start_simulator(g, tx_amt, strategy, n, graph_folder, None, churn_rate, i=interval, c=cpus, s=seed_val)
        print(f"Graph {i+1}: Simulation took {time.perf_counter() - start} seconds")


if __name__ == '__main__':
    list_of_strategies = ['random', 'highest_degree', 'k-center_gon_deg', 'k-center_gon', 'k-median', 'k-median_deg',
                          'k-median_par', 'k-median_par_deg', 'bc', 'bc_approx_10', 'bc_approx_50', 'greedy_mbi',
                          'greedy_mbi_par']
    if len(sys.argv) > 1:
        m = str(sys.argv[1])
        if m in list_of_strategies:
            if len(sys.argv) > 2:
                churn = float(sys.argv[2])
                if len(sys.argv) > 3:
                    amt = int(sys.argv[3])
                    if amt in [100, 10000, 1000000]:
                        if len(sys.argv) > 4:
                            i = int(sys.argv[4])
                            if len(sys.argv) > 5:
                                num = int(sys.argv[5])
                                if len(sys.argv) > 6:
                                    c = int(sys.argv[6])
                                    if len(sys.argv) > 7:
                                        s = int(sys.argv[7])
                                        start_eval(strategy=m, churn_rate=churn, tx_amt=amt, interval=i, n=num, cpus=c, seed_val=s)
                                    else:
                                        start_eval(strategy=m, churn_rate=churn, tx_amt=amt, interval=i, n=num, cpus=c)
                                else:
                                    start_eval(strategy=m, churn_rate=churn, tx_amt=amt, interval=i, n=num)
                            else:
                                start_eval(strategy=m, churn_rate=churn, tx_amt=amt, interval=i)
                        else:
                            start_eval(strategy=m, churn_rate=churn, tx_amt=amt)
                    else:
                        print("Warning, tx_amt is not in [100, 10000, 1000000]!")
                else:
                    start_eval(strategy=m, churn_rate=churn)
            else: 
                start_eval(strategy=m)
        else:
            print("No valid strategy given!")
    else:
        print("Parameters: <metric> <churn_rate> <tx_amt> <eval_interval> <nodes_to_add> <cpus> <seed>")
        print("Defaults (if no parameters are given): strategy='random', churn_rate= 0.956, tx_amt=100, interval=1000, n=10000")
        start_eval()
