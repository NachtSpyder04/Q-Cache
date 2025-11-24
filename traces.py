import numpy as np

# Synthetic trace generation functions, need a fix random seed?
# todo : use real traces later, refer DRLCache repo

def zipf_trace(length: int = 2000, n_items: int = 100, alpha: float = 1.5) -> np.ndarray:
    return (np.random.zipf(alpha, length) - 1) % n_items

def loop_trace(length: int = 2000, loop_size: int = 20, n_items: int = 100) -> np.ndarray:
    pattern = list(range(min(loop_size, n_items)))
    trace = (pattern * (length // len(pattern) + 1))[:length]
    return np.array(trace)


def markov_trace(length=2000, n_items=100):
    trace = []
    current = np.random.randint(0, n_items)

    for _ in range(length):
        trace.append(current)
        if np.random.rand() < 0.7:  
            current = (current + np.random.randint(-2, 3)) % n_items
        else:                       
            current = np.random.randint(0, n_items)

    return np.array(trace)


def bursty_trace(length=2000, n_items=100, burst_len=50):

    trace = []

    while len(trace) < length:
        hot = np.random.randint(0, n_items)
        trace += [hot] * burst_len
        trace += list(np.random.randint(0, n_items, burst_len))

    return np.array(trace[:length])

def stride_trace(length=2000, stride=3, n_items=100):

    trace = [(i * stride) % n_items for i in range(length)]
    return np.array(trace)

def mixed_trace(length=2000, n_items=100):

    # hv to adjust percentages to simulate real traces
    zipf_part   = (np.random.zipf(1.3, int(0.25*length)) - 1) % n_items
    loop_part   = loop_trace(int(0.20*length), 20, n_items)
    burst_part  = bursty_trace(int(0.20*length), n_items)
    random_part = np.random.randint(0, n_items, int(0.1*length))
    markov_part  = markov_trace(int(0.15*length), n_items)
    stride_part  = stride_trace(int(0.1*length), n_items)

    mixed = np.concatenate([zipf_part, loop_part, burst_part, random_part, markov_part, stride_part])
    np.random.shuffle(mixed)

    return mixed


traces = {
    'zipf': zipf_trace(length=2000, n_items=100, alpha=1.5),
    'loop': loop_trace(length=2000, loop_size=20, n_items=100),
    'markov': markov_trace(length=2000, n_items=100),
    'bursty': bursty_trace(length=2000, n_items=100, burst_len=50),
    'stride': stride_trace(length=2000, stride=3, n_items=100),

    'mixed': mixed_trace(length=2000, n_items=100)
}

print("Generated synthetic traces")
print(f"Zipf trace: {len(traces['zipf'])} requests, unique trace: {len(np.unique(traces['zipf']))}")
print(f"Loop trace: {len(traces['loop'])} requests, unique trace: {len(np.unique(traces['loop']))}")
print(f"Markov trace: {len(traces['markov'])} requests, unique trace: {len(np.unique(traces['markov']))}")
print(f"Bursty trace: {len(traces['bursty'])} requests, unique trace: {len(np.unique(traces['bursty']))}")
print(f"Stride trace: {len(traces['stride'])} requests, unique trace: {len(np.unique(traces['stride']))}")
print(f"Mixed trace: {len(traces['mixed'])} requests, unique trace: {len(np.unique(traces['mixed']))}")