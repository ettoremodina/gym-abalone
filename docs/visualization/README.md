# Game State Exploration Visualization

This module provides tools to visualize and analyze game state exploration patterns across multiple playthroughs as a tree structure.

## Overview

The `GameStateExplorer` class builds a tree-based graph where:
- **Nodes** represent unique game states (sized by visit frequency)
- **Edges** represent state transitions (thickness by transition frequency)
- **Layout** displays states left-to-right by depth (move number)
- **Branches** show where different games diverge from common paths
- **Colors** indicate different metrics (depth, player, visit count, etc.)

## Key Concept: Tree Structure

Since games rarely return to the same board state:
- **Single game** = Straight line from left to right
- **Multiple games** = Tree with branches showing where games diverge
- States at the same depth (move number) are vertically aligned
- Shared states (visited by multiple games) appear larger

## Features

- Load multiple game histories from JSON files
- Build state transition graphs with depth tracking
- Compute tree and hierarchical layouts
- Generate multiple visualization perspectives
- Analyze state revisit patterns and branching points

## Usage

### Quick Demo

See the tree structure with a simple demo:

```cmd
# Generate a few games
python examples\generate_multiple_games.py 5

# View tree visualization
python examples\demo_tree_visualization.py
```

This creates:
- Single game view (straight line)
- Multiple games view (branching tree)

### 1. Generate Game Data

Generate multiple games with history tracking:

```cmd
python examples\generate_multiple_games.py 20
```

This creates 20 games in the `data/` folder with full state history.

### 2. Install Visualization Dependencies

```cmd
pip install matplotlib networkx
```

### 3. Visualize Exploration

```cmd
python examples\visualize_exploration.py
```

This generates tree-based visualizations showing:
- State exploration tree by depth (move number)
- State exploration tree by visit count
- State exploration tree by number of games
- State exploration tree by player

## API Reference

### GameStateExplorer

```python
from gym_abalone.game.visualization.state_explorer import GameStateExplorer

explorer = GameStateExplorer(data_folder='data')
explorer.load_games(pattern='game_history_*.json')
explorer.build_state_graph(max_states=500)
explorer.print_statistics()
explorer.visualize(
    layout='tree',  # Tree layout for left-to-right visualization
    color_by='visits',
    figsize=(28, 14),
    save_path='exploration_tree.png'
)
```

### Key Methods

- **`load_games(pattern)`** - Load game files matching pattern
- **`build_state_graph(max_states)`** - Build the state transition graph with depth tracking
- **`visualize(...)`** - Create and save tree visualization
- **`print_statistics()`** - Display exploration statistics
- **`get_state_details(hash)`** - Get details about specific state

### Visualization Parameters

- **layout**: Graph layout algorithm
  - `'tree'` - Tree layout with depth-based positioning (default, recommended)
  - `'hierarchical'` - Tree layout optimized to minimize edge crossings
  - `'spring'` - Force-directed layout
  - `'kamada_kawai'` - Kamada-Kawai layout
  - `'circular'` - Circular layout
  - `'spectral'` - Spectral layout

- **color_by**: Node coloring criterion
  - `'depth'` - Color by move number/depth (shows progression)
  - `'games'` - Color by number of games visiting (shows convergence)
  - `'visits'` - Color by visit frequency (shows hotspots)
  - `'player'` - Color by player (shows turn patterns)
  - `'turn'` - Color by turn number

- **node_size_scale**: Scaling factor for node sizes
- **edge_width_scale**: Scaling factor for edge widths
- **show_labels**: Show visit counts as labels

## Statistics Provided

- Total unique states discovered
- State visit distribution
- Most visited states
- Most common transitions
- Graph connectivity metrics
- Per-state game coverage

## Example Output

```
STATE EXPLORATION STATISTICS
======================================================================

Games analyzed: 20
Unique states discovered: 1247
Total state visits: 2891
Unique transitions: 2634
Total transitions: 2671

Average visits per state: 2.32

Top 10 most visited states:
   1. State a3f2c1e4... - 45 visits, 18 games, turn 5
   2. State b7e9d3a1... - 38 visits, 15 games, turn 12
   ...

Graph connectivity:
  Average node degree: 4.28
  Graph density: 0.0034
```

## Advanced Usage

### Custom Analysis

```python
explorer = GameStateExplorer('data')
explorer.load_games()
explorer.build_state_graph()

# Get most visited state
most_visited_hash = explorer.state_visits.most_common(1)[0][0]
details = explorer.get_state_details(most_visited_hash)

print(f"State {most_visited_hash}:")
print(f"  Visits: {details['visits']}")
print(f"  Games: {len(details['metadata']['games_visited'])}")
print(f"  Board:\n{np.array(details['board'])}")
```

### Filter by Game Type

```python
# Load only specific games
explorer.load_games(pattern='game_*_classical_*.json')
```

### Analyze Convergence

```python
# Compare early vs late game states
explorer.build_state_graph()

early_states = [h for h, meta in explorer.state_metadata.items() 
                if meta['turn'] < 50]
late_states = [h for h, meta in explorer.state_metadata.items() 
               if meta['turn'] >= 50]

print(f"Early game unique states: {len(early_states)}")
print(f"Late game unique states: {len(late_states)}")
```

## Performance Notes

- For large datasets (>50 games), use `max_states` parameter to limit graph size
- Hashing uses MD5 for fast state comparison
- Graph building is O(n*m) where n=games, m=moves per game
- Visualization complexity depends on number of nodes/edges

## Troubleshooting

**Issue: Graph too dense to visualize**
- Solution: Reduce `max_states` parameter
- Solution: Use `'kamada_kawai'` or `'spectral'` layout

**Issue: Out of memory**
- Solution: Process games in batches
- Solution: Increase `max_states` threshold

**Issue: Slow visualization**
- Solution: Reduce `node_size_scale` and `edge_width_scale`
- Solution: Set `show_labels=False`
