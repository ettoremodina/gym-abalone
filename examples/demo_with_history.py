"""
Demo of AI playing with history tracking
"""
import gym
import random
from gym_abalone.envs.abalone_env import AbaloneEnv
import numpy as np

class RandomAgent:

    @staticmethod
    def choice_prioritize_random(env):
        player = env.game.current_player
        possible_moves = env.game.get_possible_moves(player, group_by_type=True)

        for move_type in ['winner', 'ejected', 'inline_push', 'sidestep_move', 'inline_move']:
            if possible_moves[move_type]:
                i_random = np.random.randint(len(possible_moves[move_type]))
                pos0, pos1 = possible_moves[move_type][i_random]
                break

        return (pos0, pos1)


env = AbaloneEnv(render_mode='terminal')

print("Playing game with history tracking enabled...")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}\n")

NB_EPISODES = 1
for episode in range(1, NB_EPISODES+1):
    env.game.enable_history_tracking(True)
    env.reset(random_player=True, random_pick=True)
    env.game.clear_history()
    # Record initial state
    env.game._record_initial_state()
    
    done = False
    while not done:
        action = RandomAgent.choice_prioritize_random(env)
        obs, reward, done, info = env.step(action)
        print(f"Turn {info['turn']:3d} | {info['player_name']:5s} | {str(info['move_type']):>16s} | reward={reward:>4.1f}")
    
    print(f"\nEpisode {episode} finished after {env.game.turns_count} turns")
    print(f"Game over: {env.game.game_over}")
    print(f"Damages: {env.game.players_damages}")
    
    history = env.game.get_history()
    print(f"\nTotal moves recorded in history: {len(history)}")
    
    saved_path = env.game.save_game_history(filepath=f'data/game_history_episode_{episode}.json')
    print(f"Game history saved to: {saved_path}")
    
    print("\nFirst 3 moves from history:")
    for i, move in enumerate(history[:3]):
        print(f"  Turn {move['turn']}: Player {move['player']} -> {move['action']['type']}")
    
    print("\nFinal game state:")
    final_state = env.game.get_state()
    print(f"  Turn: {final_state['turn']}")
    print(f"  Current player: {final_state['current_player']}")
    print(f"  Damages: {final_state['players_damages']}")
    print(f"  Game over: {final_state['game_over']}")

env.close()
print("\nDone!")
