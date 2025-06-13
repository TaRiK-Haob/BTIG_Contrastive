import random

def subgraph_sampling(line, max_nodes, rate):
    length = len(line['pkt_len_seq'])

    if length <= 6:
        return line

    if length > max_nodes:
        line['pkt_len_seq'] = line['pkt_len_seq'][:max_nodes]
        line['pkt_ts_seq'] = line['pkt_ts_seq'][:max_nodes]


    line['pkt_len_seq'] = line['pkt_len_seq'][:int(length * rate)]
    line['pkt_ts_seq'] = line['pkt_ts_seq'][:int(length * rate)]

    return line