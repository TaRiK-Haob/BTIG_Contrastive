import random

def TrafficFlowAugmentation(flow, probability=0.5):
    pkt_len_seq = flow['pkt_len_seq']
    pkt_ts_seq = flow['pkt_ts_seq']
    pkt_iat_seq = [pkt_ts_seq[i] - pkt_ts_seq[i-1] for i in range(1, len(pkt_ts_seq))]
    pkt_iat_seq.insert(0, 0)
    result = flow.copy()

    aug_len_seq = pkt_len_seq[:4]
    aug_iat_seq = pkt_iat_seq[:4]
    
    for i in range(4,len(pkt_len_seq)):
        q = random.uniform(0, 1)
        if q <= 0.5:
            #生成包
            z = random.randint(40, 1500)
            a = random.uniform(0, 0.2)
            d = random.choice([-1, 1])

            # Insert the augmented packet
            aug_len_seq.append(d * z)
            aug_iat_seq.append(a)

        r = random.uniform(0, 1)
        if r <= 0.5:
            theta = random.uniform(0, 0.2)
            pkt_iat_seq[i] += theta
            
        # Insert the original packet i
        aug_len_seq.append(pkt_len_seq[i])
        aug_iat_seq.append(pkt_iat_seq[i])

    # Calculate the cumulative timestamps
    aug_ts_seq = [0.0]
    for i in range(1, len(aug_iat_seq)):
        aug_ts_seq.append(aug_ts_seq[i-1] + aug_iat_seq[i])

    result['pkt_len_seq'] = aug_len_seq
    result['pkt_ts_seq'] = aug_ts_seq

    return result
    
if __name__ == '__main__':
    # Example usage
    flow = {"pkt_len_seq": [52, -48, 40, 557, -40, -1500, -1500, -2306, 40, 166, 93, 96, 82, 887, -298, 40, -40, -78, 78, -40, -1500, -1500, -1157, -376, 40, 40, -40, 40], "pkt_ts_seq": [0.0, 0.0288, 0.0289, 0.029, 0.0578, 0.0599, 0.0602, 0.0603, 0.0604, 0.0623, 0.0627, 0.0627, 0.0627, 0.0629, 0.0911, 0.0913, 0.0915, 0.0915, 0.0915, 0.1203, 0.1299, 0.1302, 0.1302, 0.1302, 0.1303, 2.6747, 2.7035, 2.7036], "proto": 6, "label": 0, "class": 0}

    augmented_flow = TrafficFlowAugmentation(flow, probability=0.5)
    print(augmented_flow)
    augmented_flow = TrafficFlowAugmentation(flow, probability=0.5)
    print(augmented_flow)