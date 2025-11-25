# Q-Cache
Q-Cache - A Deep Q-learning guided Cache replacement policy 

## Overview 

Q-Cache is a Deep Reinforcement Learning–based cache replacement policy that learns how to evict cache blocks more intelligently than traditional heuristics such as LRU, LFU, and Random.
Using a Deep Q-Network (DQN), the agent observes the cache state, predicts the long-term value of keeping each item, and chooses an eviction action that maximizes future hit rate.

This project evaluates Q-Cache across multiple synthetic access patterns such as Zipf, Loop, Bursty, Markov, Stride, and Mixed traces.

## Workload Patterns 

This project includes 6 different trace generators. These are :-
1. **Zipf** - Power-law distribution (models web caching, file systems)
2. **Loop** - Sequential repeated pattern (models streaming, scan workloads)
3. **Markov** - State-based transitions (models database queries)
4. **Bursty** - Temporal bursts of hot items (models social media, news)
5. **Stride** - Strided access pattern (models scientific computing)
6. **Mixed** - Composite of all patterns (models realistic workloads) 

## Why DQN instead of other Deep Learning Models?

Cache replacement is a **sequential decision-making problem**, not a classification task.

DQN was chosen because:
- It learns actions, not predictions.
- It optimizes long-term hit rate via Bellman updates.
- It handles exploration vs exploitation.
- It adapts to workload changes, unlike fixed heuristics.
- CNNs/MLPs/LSTMs cannot directly choose eviction actions.

## Workflow 

Q-Cache works in following way:- 

1. Environment: The `CacheEnv` class (in `dqn_cache.py`)simulates:

    - Cache contents
    - Recency of each item
    - Access frequency
    - State vector construction
    - Reward signal
    - Eviction and insertion behavior

2. State Representation:- The agent receives a numerical state vector composed of:

    - Cache one-hot encoding (which items are present in cache)
    - Access frequencies of all items
    - Recency positions of cached items
    - One-hot vector of the current request

    This is a 400-dimensional state vector when using 100 items.

3. Deep Q-Network (DQN) :- The DQN predicts Q-values representing: “How good or bad would it be to keep each item in cache?” The agent evicts the item with the lowest Q-value.

4. Training: The agent is trained on a composite trace (Zipf + Loop + Bursty + Markov + Stride + Random).

5. Evaluation: After training, the learned policy is evaluated on each workload and compared against:

    - LRU
    - LFU
    - Random

## Installing Dependencies 
This Project requires following dependencies :- Pytorch, NumPy, Matplotlib. To install these, execute the given command - 
```sh
pip install -r requirements.txt
```


## Instructions 
To train the DQN agent: 
```sh
python3 train.py 
```

Note: Episodes can be increased from `train.py` and hyperparameters of the model can be tuned from `dqn_cache.py` 

To evaluate/ view results: 
```sh 
python3 evaluate.py
```

## Results 
Because the access traces are randomly generated each time, the exact hit rates vary slightly between runs. However, the following trends are consistently observed:
- **ZIPF**: DQN performs better than LRU, while LFU remains the best.
- **LOOP**: All policies achieve near-perfect hit rates (~0.99).
- **MARKOV**: DQN underperforms compared to LRU due to strong temporal locality.
- **BURSTY**: LRU and LFU outperform DQN on burst-heavy patterns.
- **STRIDE**: DQN significantly outperforms all heuristic policies.
- **MIXED**: DQN performs better than LRU and remains close to LFU.

## TODO / Future Work 
 
The following project was proposed as a proof of concept as a 'Deep Learning' coursework project. However I want to take it as a serious research project in the intersected area of Artificial Intelligence and Computer Systems. Following are the few Bottlenecks that I have identified that need to be addressed to improve performance:- 

  - [ ] Improve Cache State and Action Space Representation:
        The current state is sparse and scales with total item count, not cache size, making DQN hard to train. A more compact, cache-slot–based state and action representation is needed.

  - [ ] Redesign the Reward Function:
        Simple hit/miss rewards (+1/-1) ignore write-back costs, eviction penalties, and long-term reuse. A better multi-component reward function can better reflect real cache behavior.

  - [ ] Train on Real-World Traces:
        Synthetic traces are useful for testing, but real-world traces (SPEC, Google cluster, Linux page-cache traces, etc) are necessary for research-grade evaluation.

  - [ ] Intergrate with Hardware Level Simulators ( Medium Priority ):
        Q-Cache can be intergrated into ChampSim or gem5. This allows evaluation under realistic CPU behaviour


**If I missed some critical bottlenecks or issues need to be addressed, feel free to reach out to me!**

## References 

- [DRLCache Repository](https://github.com/peihaowang/DRLCache/tree/master)
- [Reinforcement Learning, Sutton and Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
- [Deep Q-Networks Explained](https://youtu.be/x83WmvbRa2I?si=0v28YIqBpm17JRZ4)