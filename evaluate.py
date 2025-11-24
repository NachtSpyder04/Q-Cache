import numpy as np 
from dqn_cache import CacheEnv, DQNAgent, LFUPolicy, LRUPolicy, RandomPolicy
import traces
import matplotlib.pyplot as plt

def evaluate_all_policies(trace, cache_size: int = 16, num_items: int = 100, model_path: str = 'dqn_model.pt'):

    results = {}

    state_size = num_items * 4
    dqn_agent = DQNAgent(state_size, num_items)
    dqn_agent.load(model_path)


    if dqn_agent is not None:
        env = CacheEnv(cache_size=cache_size, num_items=num_items)
        env.reset()
        hits = 0
        dqn_agent.epsilon = 0 

        for request in trace:
            state = env.get_state(request)
            _, reward, hit = env.step(request)

            if hit:
                hits += 1
            else:
                if len(env.cache) > cache_size:
                    evict_idx = dqn_agent.choose_eviction(env, state)
                    env.cache.pop(evict_idx)

        results['DQN'] = hits / len(trace)

    
    env = CacheEnv(cache_size=cache_size, num_items=num_items)
    env.reset()
    lru = LRUPolicy()
    hits = 0

    for request in trace:
        _, reward, hit = env.step(request)
        if hit:
            hits += 1
        else:
            if len(env.cache) >= cache_size:
                evict_idx = lru.choose_eviction(env.cache)
                env.cache.pop(evict_idx)

    results['LRU'] = hits / len(trace)

    
    env = CacheEnv(cache_size=cache_size, num_items=num_items)
    env.reset()
    lfu = LFUPolicy()
    hits = 0

    for request in trace:
        _, reward, hit = env.step(request)
        if hit:
            hits += 1
        else:
            if len(env.cache) >= cache_size:
                evict_idx = lfu.choose_eviction(env.cache, env.access_counts)
                env.cache.pop(evict_idx)

    results['LFU'] = hits / len(trace)

    
    env = CacheEnv(cache_size=cache_size, num_items=num_items)
    env.reset()
    rand_policy = RandomPolicy()
    hits = 0

    for request in trace:
        _, reward, hit = env.step(request)
        if hit:
            hits += 1
        else:
            if len(env.cache) >= cache_size:
                evict_idx = rand_policy.choose_eviction(env.cache)
                env.cache.pop(evict_idx)

    results['Random'] = hits / len(trace)

    return results

if __name__ == "__main__":
    
    for trace_name, trace in traces.traces.items():
        print(f"\n{trace_name.upper()} Trace:")
        print("-" * 40)

        results = evaluate_all_policies(trace, cache_size=32, num_items=100, model_path='dqn_model.pt')

        for policy_name, hit_rate in results.items():
            print(f"{policy_name:10s}: {hit_rate:.3f}")

        #todo : improve plot formatting, uncomment if u want 
        # policy_names = list(results.keys())
        # hit_rates_final = list(results.values())
        # colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
        # bars = plt.bar(policy_names, hit_rates_final, color=colors[:len(policy_names)])
        # plt.xlabel('Cache Policies')
        # plt.ylabel('Hit Rate')
        # plt.title(f'Cache Policy Hit Rates on {trace_name.upper()} Trace')
        # plt.ylim(0, 1)
        # plt.grid(True, alpha=0.3, axis='y')

        # for bar in bars:
        #     height = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width()/2., height,
        #                     f'{height:.3f}',
        #                     ha='center', va='bottom', fontweight='bold')

        # plt.show()