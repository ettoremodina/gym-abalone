"""
Example: Visualize game state exploration across multiple playthroughs

This script demonstrates how to use the GameStateExplorer to analyze
and visualize state exploration patterns across multiple games.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_abalone.game.visualization.state_explorer import GameStateExplorer


def main():
    """Main execution function"""
    
    print("="*70)
    print("GAME STATE EXPLORATION VISUALIZER")
    print("="*70)
    
    explorer = GameStateExplorer(data_folder='data')
    
    num_games = explorer.load_games(pattern='abalone_game_*.json')
    
    if num_games == 0:
        print("\nNo games found! Please run some games first:")
        print("  python examples/demo_with_history.py")
        return
    
    print(f"\nAnalyzing {num_games} games...")
    
    # Don't limit states for tree visualization
    explorer.build_state_graph(max_states=None)
    
    explorer.print_statistics()
    
    print("\nGenerating tree visualization...")
    
    print("\nCreating: State Exploration Tree (alternating colors by player/turn)")
    explorer.visualize(
        layout='tree',
        color_by='player',
        figsize=(32, 16),
        node_size_scale=40,
        edge_width_scale=1.5,
        show_labels=False,
        save_path='data/exploration_tree.png'
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nVisualization file saved:")
    print("  - exploration_tree.png")
    print("\nThis shows:")
    print("  - Left to right: progression through game (by move depth)")
    print("  - Branches: where different games make different choices")
    print("  - Colors: alternating by player (turn order)")
    print("  - Node size: how many times that state was visited")
    
    most_visited = explorer.state_visits.most_common(1)[0]
    state_hash = most_visited[0]
    
    print(f"\nMost visited state: {state_hash}")
    details = explorer.get_state_details(state_hash)
    if details:
        print(f"  Visits: {details['visits']}")
        print(f"  Games: {len(details['metadata']['games_visited'])}")
        print(f"  Turn: {details['metadata']['turn']}")
        print(f"  Incoming transitions: {len(details['incoming_transitions'])}")
        print(f"  Outgoing transitions: {len(details['outgoing_transitions'])}")


if __name__ == "__main__":
    try:
        import matplotlib
        import networkx
        main()
    except ImportError as e:
        print(f"\nError: Missing required package - {e}")
        print("\nPlease install visualization dependencies:")
        print("  pip install matplotlib networkx")
        sys.exit(1)
