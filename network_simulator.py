"""
    discrete-event simulation of the Lightning Network
"""

import simpy  # version 3.0.11
import networkx as nx
import time
import sys
from random import choice, seed
import utils
import simulator as sim
import autopilot as ap
import graph_creation as gc
import random

# global parameter
cpus = None


def write_to_file(file, data, mode="w"):
    """
        writes 'data' to 'file' using 'mode' given
    """
    with open(file, mode, encoding='utf-8') as f:
        f.write(data)


def read_candidates_from_file(file):
    """
        tries to read the candidate list from the given file.
        if no such file exists it is created (empty)
    """
    try:
        with open(file, "r", encoding='utf-8') as f:
            c = f.read()
            c = c.split()
            c = [int(i) for i in c]
    except FileNotFoundError:
        with open(file, "w", encoding='utf-8') as f:
            f.close()
        c = []
    return list(c)


def add_node_data(file, node, g):
    """
        adds data about 'node' (its channels in 'g') to 'file'
    """
    data = '-----Channels of Node {}-----\n'.format(node)
    for neighbor in nx.neighbors(g, node):
        data += '{} --> {}: {}\n'.format(node, neighbor, g[node][neighbor])
        data += '{} --> {}: {}\n'.format(neighbor, node, g[neighbor][node])
    data += '--------------------------------\n\n'
    write_to_file(file, data, mode="a")


def choose_random_capacity(g, tx_amt):
    """
        creates a list of all edge capacities that belong to a channel with sufficient capacity
    """
    all_caps = [c for (u, v, c) in g.edges(data='capacity') if c >= (tx_amt / 2)]  # factor '2' due to 50:50 split
    return choice(all_caps)


def reset_edge_weights(g):
    """
        resets edge weights -> 50:50 split of the capacity
    """
    for (u, v, i) in g.edges(keys=True):
        total = g[u][v][i]['capacity'] + g[v][u][i]['capacity']
        g[u][v][i]['capacity'] = total / 2
        g[v][u][i]['capacity'] = total / 2


def evaluate_graph_centralization(g, folder):
    # reduce graph to largest component:
    max_comp = max(nx.strongly_connected_components(g), key=len)
    g_f = g.subgraph(max_comp).copy()
    print("fee graph (reduced to largest strongly connected component) has {} nodes and {} "
          "edges".format(g_f.number_of_nodes(), g_f.number_of_edges()))
    t00 = time.time()
    print("calculating degrees...", end="")
    deg = dict(nx.degree(g_f))
    write_to_file(folder + 'degrees.txt', str(deg), "a")
    t0 = time.time()
    print("took {} seconds.".format(t0 - t00))
    print("calculating betweenness centrality...", end="")
    bc = nx.betweenness_centrality(g_f, weight='fee')
    # bc = utils.shortest_path_vertex_betweenness(g_f, True, 'fee')
    write_to_file(folder + 'betweenness_centrality.txt', str(bc), "a")
    t1 = time.time()
    print("took {} seconds.".format(t1 - t0))
    # print("took {} seconds.\ncalculating closeness centrality...".format(t1 - t0), end="")
    # cc = nx.closeness_centrality(g_f, distance='fee')  # unweighted??
    # write_to_file(folder + 'closeness_centrality.txt', str(cc), "a")
    t2 = time.time()
    # print("took {} seconds.\ncalculating clustering coefficient...".format(t2 - t1), end="")
    print("calculating clustering coefficient...", end="")
    cce = nx.clustering(g_f)
    write_to_file(folder + 'clustering.txt', str(cce), "a")
    t3 = time.time()
    print("took {} seconds.\ncalculating transitivity...".format(t3 - t2), end="")
    tran = nx.transitivity(g_f)
    write_to_file(folder + 'transitivity.txt', str(tran) + "\n", "a")
    t4 = time.time()
    print("took {} seconds.\ncalculating diameter...".format(t4 - t3), end="")
    diam = nx.diameter(g_f)
    write_to_file(folder + 'diameter.txt', str(diam) + "\n", "a")
    print('took {} seconds.'.format(time.time() - t4))


class Network(object):

    def __init__(self, e, g, strategy, k, tx_amt, f, ticks, scheduler, interval, churn_rate):
        print("Init Network Model...", end="")
        self.churn_rate = churn_rate # the churn rate  e.g., 0.956 means 95.6 nodes leave per 100 added
        self.nodes_added = 0 ###### track the nodes added 
        self.env = e  # DES environment
        self.ticks = ticks  # total number of ticks to be executed
        self.g = g.copy()  # network graph
        self.g_init = g.copy()  # network graph -> used to reset 'g'
        self.g_f = gc.directed_fee_graph(g, tx_amt, exclude_edges=True, multi_graph=False)  # fee graph derived from g
        # reduce graph to largest component:
        max_comp = max(nx.strongly_connected_components(self.g_f), key=len)
        self.g_f = self.g_f.subgraph(max_comp).copy()
        print("(Fee graph has {} nodes and {} edges)".format(self.g_f.number_of_nodes(), self.g_f.number_of_edges()))
        self.strategy = strategy  # autopilot strategy to be used
        self.k = k  # channels to be opened by the autopilot
        self.tx_amt = tx_amt  # transaction amount
        self.interval = interval  # interval at which centralization is to be evaluated
        self.file = f  # file to which the results are written
        self.new_node_idx = sorted(g.nodes(), reverse=True)[0] + 1  # index of the next node to be added
        self.stats = {}  # statistics for all nodes
        self.connect_to = None  # store which nodes to connect to (-> for deterministic, comp. expensive strategies)
        for n in g.nodes():
            self.stats[n] = [0, 0, 0, 0, 0]  # [tx_attempts, tx_sent, fees_paid, tx_routed, fees_earned]
        if scheduler == 1:
            self.scheduler = e.process(self.event_scheduler1())  # start process that schedules events -> tx
        else:
            evaluate_graph_centralization(self.g_f, self.file)
            self.scheduler = e.process(self.event_scheduler2())  # start process that schedules events -> centralization

    def event_scheduler1(self):
        """
            loop scheduling and executing the next event
                => focuses on egoistic view of one new node joining the network!
        """
        print ("event scheduler 1 is running")
        while True:
            if self.env.now == 1:
                res = self.add_node(candidates=self.k)
                write_to_file(self.file, res, mode="a")
                # add_node_data(self.file, self.new_node_idx - 1, self.g)
            if self.env.now <= self.ticks / 2:
                # first half of all transactions: random source, random target
                res = self.send_tx()
            else:
                # second half of all transactions: new node as source, random target
                res = self.send_tx(s=self.new_node_idx-1)
            write_to_file(self.file, res, mode="a")
            if self.env.now == self.ticks:
                print("")
            if self.env.now == self.ticks or self.env.now == self.ticks / 2:
                res = "\n---------------STATISTICS [tx_attempts, tx_sent, fees_paid, tx_routed, fees_earned]" +\
                      "---------------\n"
                res += "(statistics of nodes are not listed if they equal [0,0,0,0,0])\n"
                total, successes = 0, 0
                info = str(self.stats[self.new_node_idx - 1]) + " "
                for n in self.g.nodes():
                    if self.stats[n] != [0, 0, 0, 0, 0]:
                        res += '{:5}'.format(str(n)) + str(self.stats[n]) + "\n"
                        total += self.stats[n][0]
                        successes += self.stats[n][1]
                        self.stats[n] = [0, 0, 0, 0, 0]  # reset statistics
                reset_edge_weights(self.g)  # reset graph
                print("reset graph and stats...")
                res += "\n"
                write_to_file(self.file, res, mode="a")
                add_node_data(self.file, self.new_node_idx - 1, self.g)

                # store important info separately:
                info += str(successes) + "/" + str(total) + "\n"
                if self.file[-6:-5] == "_":
                    write_to_file(self.file[:-5] + "stats.txt", info, "a")
                else:
                    write_to_file(self.file[:-6] + "stats.txt", info, "a")

            yield self.env.timeout(1)  # wait until next 'tick' before executing next event

    def event_scheduler2(self):
        """
            loop scheduling and executing the next event
                => focuses on many nodes joining the network and evaluating the effects on centralization
        """
        
        # initial evaluation at time 0 before nodes added
        print("Initial evaluation at time 0")
        sim_graph = self.g.copy()
        stats = sim.simulate_transactions(sim_graph, self.tx_amt, 1000,
                                          file=self.file + 'tx_0.txt')
        successes = sum([stats[i][1] for i in stats])
        success_rate = successes / 1000 if successes > 0 else 0
        avg_fees = sum([stats[i][2] for i in stats]) / successes if successes > 0 else 0
        write_to_file(self.file + 'stats.txt',
                      "Now:{}\tSuccess Rate:{}\tAverage Fees:{}\n".format(0, success_rate, avg_fees),
                      mode="a")
        
 
        while True:
            res = self.add_node2()
            write_to_file(self.file + 'results.txt', res, mode="a")
            # if self.env.now in [1, 5, 10, 20, 50, 100, 1000, 5000, 10000]:
            if self.env.now % self.interval == 0:
                #self.remove_random_node()   
                if self.churn_rate is not None:
                    num_to_remove = int(self.nodes_added * self.churn_rate) # Remove 95 random nodes every 100 nodes added if the churn rate is 0.95
                    print( num_to_remove , " nodes will be removed")
                    self.nodes_added = 0  # reset after processing
                    self.remove_random_node(num_to_remove)
                self.g_f = gc.directed_fee_graph(self.g, self.tx_amt, exclude_edges=True, multi_graph=False)
                max_comp = max(nx.strongly_connected_components(self.g_f), key=len)
                self.g_f = self.g_f.subgraph(max_comp).copy()
                print("evaluate graph centralization...")
                evaluate_graph_centralization(self.g_f, self.file)
                # test transaction success rate in the network:
                sim_graph = self.g.copy()
                stats = sim.simulate_transactions(sim_graph, self.tx_amt, 1000,
                                                  file=self.file + 'tx_{}.txt'.format(self.env.now))
                successes = sum([stats[i][1] for i in stats])
                success_rate = successes / 1000
                avg_fees = sum([stats[i][2] for i in stats]) / successes
                write_to_file(self.file + 'stats.txt',
                              "Now:{}\tSuccess Rate:{}\tAverage Fees:{}\n".format(self.env.now, success_rate, avg_fees),
                              mode="a")
                


            yield self.env.timeout(1)  # wait until next 'tick' before executing next event

    def send_tx(self, s=None):
        """
            sends transaction with random value from random source to random target in the network
            Output: T [source] --[tx_amt]--> [target] : S/F [additional info]
                    => T: tx identifier, S: success, F: failure
        """
        sys.stdout.write("\rNOW:" + str(self.env.now))
        source = choice(list(self.g.nodes()))  # source of transaction -> not necessarily in largest conn. component
        # source = choice(list(self.g_f.nodes()))  # source of transaction
        if s is not None:
            source = s
        target = choice(list(set(self.g.nodes()) - {source}))  # recipient of transaction -> see above
        # target = choice(list(set(self.g_f.nodes()) - {source}))  # recipient of transaction
        res = 'T {:4} --{}--> {:4} : '.format(source, self.tx_amt, target)

        self.stats[source][0] += 1  # 'source' attempts to send a tx

        # test if direct channel (with sufficient capacity) exists:
        if self.g.has_edge(source, target):
            tx_failed = 1
            for j in self.g[source][target]:  # loop through all channels between 'source' and 'target'
                if not self.g[source][target][j]['disabled']:
                    if self.g[source][target][j]['capacity'] >= self.tx_amt:  # ('source' knows its balance shares)
                        tx_failed = sim.send_a_to_b(self.g, source, target, self.tx_amt)
                        if not tx_failed:
                            self.stats[source][1] += 1
                            res += "S (direct channel)\n"
                            # print(self.tx_amt, "sat were successfully sent from", source, "to", target)
                        break
            if not tx_failed:
                return res

        # a faster option here would be to simply choose the shortest (=cheapest) path in the fee graph, but as
        # outlined in the thesis, this is not always accurate (e.g. first channel on route does not charge fees)

        # do routing on graph 'g':
        path = sim.routing(self.g, source, target, self.tx_amt)

        if not path:
            # print("Failure (no sufficiently funded route exists)")
            res += "F (no sufficiently funded route exists)\n"
            return res
        failed, fees = sim.route_tx(self.g, path, self.tx_amt)
        if failed:
            # route exists, but routing failed as exact shares of channel balances are unknown
            res += "F (temporary channel failure)\n"
            # print("Transaction failed (", self.tx_amt, "from", source, "to", target, ")")
        else:
            self.stats[source][1] += 1
            self.stats[source][2] += abs(fees[source])
            for j in range(1, len(fees)):
                self.stats[path[j][0]][3] += 1  # this node successfully routed a tx
                self.stats[path[j][0]][4] += fees[path[j][0]]  # this node earned routing fees
            res += "S (Route: {})\n".format(path)
            # print(self.tx_amt, "sat were successfully sent from", source, "to", target)

        return res

    def add_node(self, candidates=None):
        """
            adds new node to the network ~> connects it using the autopilot
            Output: J [ID]
                    => J: 'join' operation (new node added to the network)
            ~> used in event_scheduler1
        """
        print('NOW:', self.env.now)
        self.g.add_node(self.new_node_idx, key='connected_using_autopilot', alias='new_node')
        self.g_f.add_node(self.new_node_idx, key='connected_using_autopilot', alias='new_node')

        k = 10
        if candidates:
            k = candidates
        w_val = (self.tx_amt * (1 / 10 ** 6) + 1) * 1000  # DEFAULT SETTINGS
        if (self.strategy == 'bc' or self.strategy == 'bc_approx_10'  or self.strategy == 'greedy_mbi' or self.strategy == 'highest_degree' or  self.strategy == 'asp' or self.strategy == 'gini_bc' or
                self.strategy == 'closeness' or self.strategy == 'diameter' or 
                (self.strategy == 'k-median_deg' and (self.tx_amt == 100)) or
                (self.strategy == 'k-median_degree' and (self.tx_amt == 100)) or   ###########
                (self.strategy == 'k-median' and (self.tx_amt == 100))):  
            print("reading candidates from file...", end="")
            # con = read_candidates_from_file(
            #    './simulation_results/{}/amt_{}/candidates.txt'.format(self.strategy, self.tx_amt))
            con = read_candidates_from_file(self.file.split('/k')[0] + '/candidates.txt')
            if con:
                con = con[:k]
                print("done")
            else:  # file 'candidates.txt' does not exist yet -> create and fill it
                print("failed. Calculate candidate set and write to file...")
                start = time.time()
                con = ap.start_autopilot(self.g_f, self.new_node_idx, 15, metric=self.strategy, weight='fee',
                                         weight_value=w_val)
                end = time.time()
                if self.file[-6:-5] == "_":
                    write_to_file(self.file[:-5] + "times.txt", str(end - start) + "\n", "a")
                else:
                    write_to_file(self.file[:-6] + "times.txt", str(end - start) + "\n", "a")
                for i in range(len(con)):
                    # write_to_file('./simulation_results2/{}/amt_{}/candidates.txt'.format(self.strategy, self.tx_amt),
                    #              str(con[i]) + " ", "a")
                    write_to_file(self.file.split('/k_')[0] + '/candidates.txt', str(con[i]) + " ", "a")
                con = con[:k]
        else:
            start = time.time()
            con = ap.start_autopilot(self.g_f, self.new_node_idx, k, metric=self.strategy, weight='fee',
                                     weight_value=w_val)
            end = time.time()
            if self.file[-6:-5] == "_":
                write_to_file(self.file[:-5] + "times.txt", str(end-start) + "\n", "a")
            else:
                write_to_file(self.file[:-6] + "times.txt", str(end-start) + "\n", "a")
        utils.connect(self.g, self.new_node_idx, con, cap=self.tx_amt*2000)
        self.g_f = gc.directed_fee_graph(self.g, self.tx_amt, exclude_edges=True, multi_graph=False)
        # reduce graph to largest component:
        max_comp = max(nx.strongly_connected_components(self.g_f), key=len)
        self.g_f = self.g_f.subgraph(max_comp).copy()
        print('Added node {} and connected it to {}'.format(self.new_node_idx, con))
        res = 'J {:4} (opened channels to {})\n'.format(self.new_node_idx, con)
        self.stats[self.new_node_idx] = [0, 0, 0, 0, 0]
        self.new_node_idx += 1

        return res

    def add_node2(self):
        """
            adds new node to the network ~> connects it using the autopilot
            Output: J [ID]
                    => J: 'join' operation (new node added to the network)
        """
        print('NOW:', self.env.now)
        self.g.add_node(self.new_node_idx, key='connected_using_autopilot', alias='new_node')
        self.g_f.add_node(self.new_node_idx, key='connected_using_autopilot', alias='new_node')
        k = 10  # fixed number of channels to 10
        w_val = (self.tx_amt * (1 / 10 ** 6) + 1) * 1000  # DEFAULT SETTINGS
        con = ap.start_autopilot(self.g_f, self.new_node_idx, k, metric=self.strategy, weight='fee', weight_value=w_val,
                                 cpus=cpus)
        for n in con:
            cap = choose_random_capacity(self.g, self.tx_amt)
            self.g.add_edge(self.new_node_idx, n, capacity=cap, fee_rate=1, base_fee=1000, delay=144, disabled=False)
            self.g.add_edge(n, self.new_node_idx, capacity=cap, fee_rate=1, base_fee=1000, delay=144, disabled=False)
        # utils.connect(self.g, self.new_node_idx, con, cap=capacity)
        self.g_f = gc.directed_fee_graph(self.g, self.tx_amt, exclude_edges=True, multi_graph=False)
        # reduce graph to largest component:
        max_comp = max(nx.strongly_connected_components(self.g_f), key=len)
        self.g_f = self.g_f.subgraph(max_comp).copy()
        print('Added node {} and connected it to {}'.format(self.new_node_idx, con))
        res = 'J {:4} (opened channels to {})\n'.format(self.new_node_idx, con)
        self.stats[self.new_node_idx] = [0, 0, 0, 0, 0]
        self.new_node_idx += 1
        self.nodes_added += 1  # Track number of nodes added #####
            
        return res

    def remove_node(self):
        """
            removes a node from the network
            Output: L [ID]
                    => L: 'leave' operation (node left the network)
        """
        print('NOW:', self.env.now)
        node = choice(list(self.g.nodes()))  # node to be deleted from the network
        c = nx.degree(self.g, self.new_node_idx)
        self.g.remove_node(node)
        print('Removed node {} and its {} channels'.format(node, c))
        res = 'L {} ({} channels removed)\n'.format(node, c)
        return res

    ######

    def remove_random_node(self, count=475):
      
        print('NOW:', self.env.now)
        if len(self.g.nodes()) < count:
            print(f"Not enough nodes to remove. Needed: {count}, Available: {len(self.g.nodes())}")
            return

        nodes_to_remove = random.sample(list(self.g.nodes()), count)
        with open("removed_nodes.txt", "a", encoding='utf-8') as f:
            f.write(f"NOW: {self.env.now}\n")
            f.write("Nodes removed: ")
            for node in nodes_to_remove:
                if self.g.has_node(node):
                    c = self.g.degree(node)
                    self.g.remove_node(node)
                    print(f'Removed node {node} and its {c} channels')
                    res = f'L {node} ({c} channels removed)\n'
                    f.write(f"{node} ")
                else:
                    print(f"Node {node} does not exist.")
            f.write("\n")

    def disable_channel(self):
        """
            disables an active channel (in one direction)
            Output: D [Node ID 1] [Node ID 2]
                    => D: 'disable' a channel
        """
        print('NOW:', self.env.now)
        active_channels = [(u, v, k) for (u, v, k, d) in self.g.edges(data='disabled', keys=True) if not d]
        c = choice(active_channels)
        self.g[c[0]][c[1]][c[2]]['disabled'] = True
        res = 'D {}--{} (key={})'.format(c[0], c[1], c[2])
        return res

    def activate_channel(self):
        """
            activates a disabled channel (in one direction)
            Output: A [Node ID 1] [Node ID 2]
                    => A: 'activate' a channel
        """
        print('NOW:', self.env.now)
        disabled_channels = [(u, v, k) for (u, v, k, d) in self.g.edges(data='disabled', keys=True) if d]
        c = choice(disabled_channels)
        self.g[c[0]][c[1]][c[2]]['disabled'] = False
        res = 'A {}--{} (key={})'.format(c[0], c[1], c[2])

        return res

    def open_channel(self):
        """
            opens a new channel
            Output: O [Node ID 1] [Node ID 2]
        """
        print('NOW:', self.env.now)
        node1 = choice(list(self.g.nodes()))  # node opening a new channel
        node2 = choice([n for n in self.g.nodes() if not self.g.has_edge(node1, n)])  # non-neighbors of 'node1'
        utils.connect(self.g, node1, [node2], 1000000)
        res = 'O {}--{}\n'.format(node1, node2)

        return res

    def close_channel(self):
        """
            closes an existing channel
            Output: C [Node ID 1] [Node ID 2] (key=[edge index])
                    => C: 'close' an existing channel
        """
        print('NOW:', self.env.now)
        channels = [(u, v, k) for (u, v, k) in self.g.edges(keys=True)]
        c = choice(channels)
        self.g.remove_edge(c[0], c[1], key=c[2])
        self.g.remove_edge(c[1], c[0], key=c[2])
        res = 'C {}--{} (key={})\n'.format(c[0], c[1], c[2])
        return res


def start_simulator(graph, tx_amt, strategy, k, folder, file, churn_rate=0.956, i=None, c=None, s=23):
    """
        file given -> connect new node to 'k' candidates using autopilot and simulate random transactions
        folder given -> add 'k' nodes and evaluate centralization of the network every 'i' steps
    """
    env = simpy.Environment(initial_time=1)  # create DES environment

    # set seed value:
    seed_val = int(s)
    seed(seed_val)

    if folder is None:
        print ("folder is none")
        until = 2000  # -> 1000 tx with fixed source, 1000 tx with random source
        write_to_file(file, 'seed value: {}\n\n'.format(seed_val))
        Network(env, graph, strategy, k, tx_amt, file, until, 1, None, churn_rate)  # create Network Model using event scheduler 1
    else:
        global cpus
        cpus = c
        until = k  # -> adding 'k' nodes (default=10000)
        write_to_file(folder + 'results.txt', 'seed value: {}\n\n'.format(seed_val))
        params = "strategy={}\ntx_amt={}\nn={}\ninterval={}\n\n".format(strategy, tx_amt, k, i)
        write_to_file(folder + 'results.txt', params, mode='a')
        Network(env, graph, strategy, k, tx_amt, folder, until, 2, i, churn_rate)  # use event scheduler 2


    prev = -1
    while env.peek() <= until:  # peek() returns time of next scheduled event or infinity (float('inf'))
        if prev < env.now:
            # print("NOW:", env.now)
            prev += 1
        env.step()  # steps to the next scheduled event
