import numpy as np
from dqn_cache import CacheEnv, DQNAgent
import traces


def train_dqn(episodes: int = 500, trace_type: str = 'mixed', cache_size: int = 16, num_items: int = 100, save_path: str = 'dqn_model.pt'):

    env = CacheEnv(cache_size=cache_size, num_items=num_items)
    state_size = num_items * 4  # x 4 cause state is one-hot of cache + freq + recency + request (optimization required)
    agent = DQNAgent(state_size, num_items)


    trace = traces.traces[trace_type]

    hit_rates = []
    losses = []
    epsilons = []

    print(f"\n{'='*60}")
    print(f"Training DQN on {trace_type} trace")
    print(f"Cache size: {cache_size}, Items: {num_items}")
    print(f"State size: {state_size}, Action size: {num_items}")
    print(f"{'='*60}\n")

    steps = 0
    for episode in range(episodes):
        env.reset()
        hits = 0
        misses = 0
        episode_loss = []

        for request in trace:
            state = env.get_state(request)
            state_before = state.copy()

            next_state, reward, hit = env.step(request)

            if hit:
                hits += 1
            else:
                misses += 1
                
                if len(env.cache) >= cache_size:
                    
                    evict_idx = agent.choose_eviction(env, state_before)
                    evicted_item = env.cache[evict_idx]

                    
                    env.cache.pop(evict_idx)

                    
                    agent.remember(state_before, evicted_item, reward, next_state)

                    
                    steps += 1

                    #todo: need  to adjust for apt replay steps
                    if steps % 4 == 0:
                        loss = agent.replay()
                        if loss > 0:
                            episode_loss.append(loss)


        hit_rate = hits / len(trace)
        hit_rates.append(hit_rate)
        epsilons.append(agent.epsilon)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)

        
        agent.decay_epsilon()

        
        if episode % agent.target_update == 0 and episode > 0:
            agent.update_target_model()

        if episode % 100 == 0:
          print(f"Ep {episode}  HR: {hit_rate:.3f}  Loss: {avg_loss:.6f}"
                  f"  Eps: {agent.epsilon:.3f}  Hits: {hits}  Misses: {misses:4d}")
          
          
    agent.save(save_path)
    print(f"\nTraining complete")
    print(f"Final hit rate: {hit_rates[-1]:.3f}")
    print(f"Best hit rate: {max(hit_rates):.3f} at episode {hit_rates.index(max(hit_rates))}")

    return agent, hit_rates, losses, epsilons, env

if __name__ == "__main__":
    
    #TOdo: need more experimentation here 
    train_dqn(
        episodes=1000,
        trace_type='mixed',
        cache_size=32,
        num_items=100,
        save_path='dqn_model.pt'
    )   