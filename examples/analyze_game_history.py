"""
Script to load and analyze saved game history
"""
import json
import sys
from pathlib import Path

def analyze_game_history(filepath):
    """
    Load and analyze a saved game history file
    
    Args:
        filepath (str): Path to the JSON history file
    """
    with open(filepath, 'r') as f:
        game_data = json.load(f)
    
    history = game_data.get('history', [])
    metadata = game_data.get('metadata', {})
    
    print("=" * 70)
    print("GAME ANALYSIS")
    print("=" * 70)
    
    if metadata:
        print("\nMetadata:")
        print(f"  Variant: {metadata.get('variant', {}).get('id', 'Unknown')}")
        print(f"  Players: {metadata.get('players', 'Unknown')}")
        print(f"  Episode: {metadata.get('episode', 'Unknown')}")
        print(f"  Total Turns: {metadata.get('total_turns', 'Unknown')}")
        print(f"  Game Over: {metadata.get('game_over', 'Unknown')}")
        print(f"  Final Damages: {metadata.get('final_damages', 'Unknown')}")
        print(f"  Timestamp: {metadata.get('timestamp', 'Unknown')}")
    
    print(f"\nTotal Moves: {len(history)}")
    
    move_types = {}
    for move in history:
        move_type = move['action']['type']
        move_types[move_type] = move_types.get(move_type, 0) + 1
    
    print("\nMove Type Distribution:")
    for move_type, count in sorted(move_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(history)) * 100
        print(f"  {move_type:>16s}: {count:3d} ({percentage:5.1f}%)")
    
    player_moves = {}
    for move in history:
        player = move['player']
        player_moves[player] = player_moves.get(player, 0) + 1
    
    print("\nMoves per Player:")
    for player, count in sorted(player_moves.items()):
        print(f"  Player {player}: {count} moves")
    
    print("\nFirst 5 moves:")
    for i, move in enumerate(history[:5]):
        print(f"  Turn {move['turn']:3d} | Player {move['player']} | "
              f"{move['action']['from']:2d} -> {move['action']['to']:2d} | "
              f"{move['action']['type']:>16s}")
    
    print("\nLast 5 moves:")
    for move in history[-5:]:
        print(f"  Turn {move['turn']:3d} | Player {move['player']} | "
              f"{move['action']['from']:2d} -> {move['action']['to']:2d} | "
              f"{move['action']['type']:>16s}")
    
    ejections = [m for m in history if m['action']['type'] == 'ejected']
    if ejections:
        print(f"\nEjections ({len(ejections)} total):")
        for move in ejections:
            print(f"  Turn {move['turn']:3d} | Player {move['player']} ejected opponent's marble")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        files = sorted(Path('data').glob('game_history_*.json'))
        if files:
            filepath = files[-1]
            print(f"No file specified, using most recent: {filepath}\n")
        else:
            print("No history files found in data folder. Please specify a file:")
            print("  python analyze_game_history.py <path_to_history.json>")
            sys.exit(1)
    
    analyze_game_history(filepath)
