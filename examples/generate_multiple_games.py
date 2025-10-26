"""
Generate multiple games with history for exploration analysis
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


def generate_games(num_games=10, max_turns=200):
    """
    Generate multiple games with history tracking
    
    Args:
        num_games (int): Number of games to generate
        max_turns (int): Maximum turns per game
    """
    print(f"Generating {num_games} games for exploration analysis...")
    print("="*70)
    
    env = AbaloneEnv(render_mode='terminal')
    
    for episode in range(1, num_games + 1):
        print(f"\n[{episode}/{num_games}] Starting game {episode}...")
        
        # Enable history before reset to capture initial state
        env.game.enable_history_tracking(True)
        # Use same variant for all games so they truly start from same state
        env.reset(random_player=True, random_pick=False, variant_name='classical')
        env.game.clear_history()
        # Record initial state after clear
        env.game._record_initial_state()
        
        done = False
        turn = 0
        
        while not done and turn < max_turns:
            action = RandomAgent.choice_prioritize_random(env)
            obs, reward, done, info = env.step(action)
            turn += 1
            
            if turn % 50 == 0:
                print(f"  Turn {turn}...")
        
        saved_path = env.game.save_game_history()
        print(f"  Game {episode} finished after {env.game.turns_count} turns")
        print(f"  Damages: {env.game.players_damages}")
        print(f"  History saved: {saved_path}")
        print(f"  States recorded: {len(env.game.get_history())}")
    
    env.close()
    
    print("\n" + "="*70)
    print(f"GENERATION COMPLETE! {num_games} games saved in data/ folder")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    
    print(f"Configuration:")
    print(f"  Number of games: {num_games}")
    print(f"  Max turns per game: {max_turns}")
    print()
    
    generate_games(num_games=num_games, max_turns=max_turns)
