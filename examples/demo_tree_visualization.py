"""
Simple demonstration of tree-based state exploration visualization

This script generates a small number of games to clearly show the tree structure:
- Single game = straight line from left to right
- Multiple games = tree with branches showing divergence
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_abalone.game.visualization.state_explorer import GameStateExplorer
import matplotlib.pyplot as plt


def demo_single_game():
    """Demonstrate single game as a straight line"""
    print("="*70)
    print("DEMO: Single Game (Straight Line)")
    print("="*70)
    
    explorer = GameStateExplorer(data_folder='data')
    
    game_files = sorted(Path('data').glob('game_history_*.json'))
    if not game_files:
        print("\nNo games found! Please run:")
        print("  python examples/generate_multiple_games.py 5")
        return
    
    explorer.games_data = []
    with open(game_files[0], 'r') as f:
        import json
        game_data = json.load(f)
        explorer.games_data.append({
            'filepath': game_files[0],
            'history': game_data.get('history', []),
            'metadata': game_data.get('metadata', {})
        })
    
    print(f"\nLoaded 1 game with {len(explorer.games_data[0]['history'])} moves")
    
    explorer.build_state_graph()
    
    print("\nGenerating straight-line visualization...")
    explorer.visualize(
        layout='tree',
        color_by='depth',
        figsize=(28, 6),
        node_size_scale=50,
        edge_width_scale=1.0,
        show_labels=False,
        save_path='data/demo_single_game_line.png'
    )
    
    print("\n✓ Saved to: data/demo_single_game_line.png")
    print("  You should see a straight line from left to right!")


def demo_multiple_games(num_games=5):
    """Demonstrate multiple games as a branching tree"""
    print("\n" + "="*70)
    print(f"DEMO: {num_games} Games (Branching Tree)")
    print("="*70)
    
    explorer = GameStateExplorer(data_folder='data')
    
    num_loaded = explorer.load_games(pattern='game_history_*.json')
    
    if num_loaded == 0:
        print("\nNo games found! Please run:")
        print("  python examples/generate_multiple_games.py 5")
        return
    
    if num_loaded > num_games:
        print(f"\nUsing first {num_games} games out of {num_loaded} available")
        explorer.games_data = explorer.games_data[:num_games]
    
    # Build graph without limiting states to see all branches
    explorer.build_state_graph(max_states=None)
    
    explorer.print_statistics()
    
    print("\nGenerating tree visualization...")
    
    explorer.visualize(
        layout='tree',
        color_by='player',
        figsize=(32, 16),
        node_size_scale=50,
        edge_width_scale=1.5,
        show_labels=False,
        save_path='data/demo_tree_exploration.png'
    )
    print("  ✓ Saved: data/demo_tree_exploration.png")
    print("    Colors alternate by player (turn order)")
    print("    Branches show where games diverge")


if __name__ == "__main__":
    try:
        import matplotlib
        import networkx
        
        print("\n" + "="*70)
        print("STATE EXPLORATION TREE DEMO")
        print("="*70)
        print("\nThis demo shows how game state exploration looks as a tree:")
        print("  - Single game: Straight line (left to right)")
        print("  - Multiple games: Tree with branches showing divergence")
        print()
        
        demo_single_game()
        
        num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        demo_multiple_games(num_games=num_games)
        
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print("\nKey observations:")
        print("  1. Games start from common initial state (leftmost)")
        print("  2. Branches show where games make different moves")
        print("  3. Nodes shared by multiple games are larger/different color")
        print("  4. Progression is always left-to-right (increasing depth)")
        print()
        
    except ImportError as e:
        print(f"\nError: Missing required package - {e}")
        print("\nPlease install visualization dependencies:")
        print("  pip install matplotlib networkx")
        sys.exit(1)
