"""
Game State Exploration Visualizer

This module provides visualization tools for analyzing game state exploration
across multiple playthroughs. It builds a graph representation of state transitions
and visualizes how often each state is visited.
"""
import json
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Optional


class GameStateExplorer:
    """
    Visualizes exploration of game states across multiple playthroughs
    
    Creates a network graph showing:
    - Nodes: Unique game states (size = visit frequency)
    - Edges: State transitions (thickness = transition frequency)
    - Colors: Different game outcomes or depths
    """
    
    def __init__(self, data_folder: str = 'data'):
        """
        Initialize the explorer
        
        Args:
            data_folder (str): Path to folder containing game history JSON files
        """
        self.data_folder = Path(data_folder)
        self.games_data = []
        self.state_graph = nx.DiGraph()
        self.state_visits = Counter()
        self.transition_counts = defaultdict(int)
        self.state_to_hash = {}
        self.hash_to_state = {}
        self.state_metadata = {}
        
    def _hash_board_state(self, board_state: List[List[int]]) -> str:
        """
        Create a unique hash for a board state
        
        Args:
            board_state (list): 2D board representation
            
        Returns:
            str: Hash string representing the state
        """
        board_array = np.array(board_state)
        board_bytes = board_array.tobytes()
        return hashlib.md5(board_bytes).hexdigest()[:16]
    
    def load_games(self, pattern: str = 'game_history_*.json') -> int:
        """
        Load all game files from the data folder
        
        Args:
            pattern (str): Glob pattern to match game files
            
        Returns:
            int: Number of games loaded
        """
        game_files = sorted(self.data_folder.glob(pattern))
        
        if not game_files:
            print(f"No game files found in {self.data_folder}")
            return 0
        
        print(f"Loading {len(game_files)} game files...")
        
        for filepath in game_files:
            try:
                with open(filepath, 'r') as f:
                    game_data = json.load(f)
                    self.games_data.append({
                        'filepath': filepath,
                        'history': game_data.get('history', []),
                        'metadata': game_data.get('metadata', {})
                    })
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        print(f"Successfully loaded {len(self.games_data)} games")
        return len(self.games_data)
    
    def build_state_graph(self, max_states: Optional[int] = None):
        """
        Build the state transition graph from loaded games
        
        Args:
            max_states (int): Maximum number of states to include (most visited)
        """
        print("\nBuilding state graph...")
        
        all_state_hashes = set()
        transitions = []
        self.game_paths = []
        
        for game_idx, game in enumerate(self.games_data):
            history = game['history']
            prev_hash = None
            game_path = []
            
            for move_idx, move in enumerate(history):
                board_state = move['board_state']
                state_hash = self._hash_board_state(board_state)
                game_path.append(state_hash)
                
                if state_hash not in self.hash_to_state:
                    self.hash_to_state[state_hash] = board_state
                    self.state_metadata[state_hash] = {
                        'turn': move['turn'],
                        'player': move['player'],
                        'damages': move['damages'],
                        'games_visited': set([game_idx]),
                        'first_seen': (game_idx, move_idx),
                        'depth': move_idx
                    }
                else:
                    self.state_metadata[state_hash]['games_visited'].add(game_idx)
                
                self.state_visits[state_hash] += 1
                all_state_hashes.add(state_hash)
                
                if prev_hash is not None:
                    transition = (prev_hash, state_hash)
                    self.transition_counts[transition] += 1
                    transitions.append(transition)
                
                prev_hash = state_hash
            
            self.game_paths.append(game_path)
        
        # Don't filter by max_states for tree visualization - keep all states
        # to maintain complete game paths
        if max_states and max_states < len(all_state_hashes):
            print(f"  Note: Limiting to {max_states} most visited states")
            most_common_states = [h for h, _ in self.state_visits.most_common(max_states)]
            # Keep states that are part of paths to most common states
            states_to_keep = set(most_common_states)
            for path in self.game_paths:
                for i, state in enumerate(path):
                    if state in states_to_keep:
                        # Include all states before this one in the path
                        states_to_keep.update(path[:i+1])
                        break
            all_state_hashes = states_to_keep
        
        for state_hash in all_state_hashes:
            visit_count = self.state_visits[state_hash]
            metadata = self.state_metadata[state_hash]
            
            self.state_graph.add_node(
                state_hash,
                visits=visit_count,
                turn=metadata['turn'],
                player=metadata['player'],
                damages=tuple(metadata['damages']),
                num_games=len(metadata['games_visited'])
            )
        
        for (from_state, to_state), count in self.transition_counts.items():
            if from_state in all_state_hashes and to_state in all_state_hashes:
                self.state_graph.add_edge(from_state, to_state, weight=count)
        
        print(f"Graph built: {self.state_graph.number_of_nodes()} nodes, "
              f"{self.state_graph.number_of_edges()} edges")
    
    def _compute_tree_layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute tree-based layout where states are positioned left-to-right by depth
        and vertically separated when games diverge
        
        Returns:
            dict: Mapping of node -> (x, y) position
        """
        pos = {}
        
        depth_to_nodes = defaultdict(list)
        for node in self.state_graph.nodes():
            depth = self.state_metadata[node]['depth']
            depth_to_nodes[depth].append(node)
        
        max_depth = max(depth_to_nodes.keys()) if depth_to_nodes else 0
        
        node_to_y = {}
        y_counter = 0.0
        y_spacing = 1.0
        
        for depth in sorted(depth_to_nodes.keys()):
            nodes = depth_to_nodes[depth]
            
            nodes_by_parent = defaultdict(list)
            for node in nodes:
                parents = list(self.state_graph.predecessors(node))
                parent_key = tuple(sorted(parents)) if parents else ()
                nodes_by_parent[parent_key].append(node)
            
            for parent_key in sorted(nodes_by_parent.keys()):
                group_nodes = nodes_by_parent[parent_key]
                
                if parent_key:
                    parent_y_values = [node_to_y.get(p, 0) for p in parent_key]
                    center_y = sum(parent_y_values) / len(parent_y_values) if parent_y_values else y_counter
                else:
                    center_y = y_counter
                
                num_nodes_in_group = len(group_nodes)
                
                if num_nodes_in_group == 1:
                    node = group_nodes[0]
                    x = depth / max(max_depth, 1) if max_depth > 0 else 0.5
                    y = center_y
                    pos[node] = (x, y)
                    node_to_y[node] = y
                    y_counter = max(y_counter, y + y_spacing)
                else:
                    start_y = center_y - (num_nodes_in_group - 1) * y_spacing / 2
                    for i, node in enumerate(sorted(group_nodes)):
                        x = depth / max(max_depth, 1) if max_depth > 0 else 0.5
                        y = start_y + i * y_spacing
                        pos[node] = (x, y)
                        node_to_y[node] = y
                        y_counter = max(y_counter, y + y_spacing)
        
        if pos:
            y_values = [y for x, y in pos.values()]
            y_min, y_max = min(y_values), max(y_values)
            y_range = y_max - y_min if y_max > y_min else 1.0
            
            pos = {node: (x, (y - y_min) / y_range) for node, (x, y) in pos.items()}
        
        return pos
    
    def _compute_hierarchical_tree_layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute hierarchical tree layout that follows game paths
        Arranges nodes to minimize edge crossings
        
        Returns:
            dict: Mapping of node -> (x, y) position
        """
        pos = {}
        
        depth_assignments = {}
        for node in self.state_graph.nodes():
            depth_assignments[node] = self.state_metadata[node]['depth']
        
        depth_to_nodes = defaultdict(list)
        for node, depth in depth_assignments.items():
            depth_to_nodes[depth].append(node)
        
        max_depth = max(depth_to_nodes.keys()) if depth_to_nodes else 0
        
        node_to_y = {}
        y_offset = 0.0
        
        for depth in sorted(depth_to_nodes.keys()):
            nodes = depth_to_nodes[depth]
            
            nodes_with_parents = []
            for node in nodes:
                parents = list(self.state_graph.predecessors(node))
                avg_parent_y = 0.5
                if parents:
                    parent_ys = [node_to_y.get(p, 0.5) for p in parents]
                    avg_parent_y = sum(parent_ys) / len(parent_ys)
                nodes_with_parents.append((node, avg_parent_y))
            
            nodes_with_parents.sort(key=lambda x: x[1])
            
            num_nodes = len(nodes)
            for i, (node, _) in enumerate(nodes_with_parents):
                x = depth / max(max_depth, 1) if max_depth > 0 else 0.5
                y = (i + 0.5) / max(num_nodes, 1)
                pos[node] = (x, y)
                node_to_y[node] = y
        
        return pos
    
    def visualize(self, 
                  layout: str = 'tree',
                  figsize: Tuple[int, int] = (24, 12),
                  node_size_scale: float = 100,
                  edge_width_scale: float = 0.5,
                  show_labels: bool = False,
                  color_by: str = 'turn',
                  save_path: Optional[str] = None):
        """
        Visualize the state exploration graph
        
        Args:
            layout (str): Graph layout algorithm ('tree', 'hierarchical', 'spring', 'kamada_kawai', etc.)
            figsize (tuple): Figure size (width, height)
            node_size_scale (float): Scaling factor for node sizes
            edge_width_scale (float): Scaling factor for edge widths
            show_labels (bool): Whether to show node labels
            color_by (str): Node coloring criterion ('turn', 'player', 'visits', 'games', 'depth')
            save_path (str): Path to save the figure (None = display only)
        """
        if self.state_graph.number_of_nodes() == 0:
            print("No graph to visualize. Build the graph first.")
            return
        
        print(f"\nVisualizing graph with {layout} layout...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if layout == 'tree':
            pos = self._compute_tree_layout()
        elif layout == 'hierarchical':
            pos = self._compute_hierarchical_tree_layout()
        elif layout == 'spring':
            pos = nx.spring_layout(self.state_graph, k=2, iterations=50, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.state_graph)
        elif layout == 'circular':
            pos = nx.circular_layout(self.state_graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.state_graph)
        else:
            pos = self._compute_tree_layout()
        
        node_sizes = [self.state_graph.nodes[node]['visits'] * node_size_scale 
                      for node in self.state_graph.nodes()]
        
        if color_by == 'turn':
            node_colors = [self.state_graph.nodes[node]['turn'] 
                          for node in self.state_graph.nodes()]
            cmap = 'viridis'
            label = 'Turn Number'
        elif color_by == 'player':
            node_colors = [self.state_graph.nodes[node]['player'] 
                          for node in self.state_graph.nodes()]
            cmap = 'RdYlBu'
            label = 'Player'
        elif color_by == 'visits':
            node_colors = [self.state_graph.nodes[node]['visits'] 
                          for node in self.state_graph.nodes()]
            cmap = 'hot'
            label = 'Visit Count'
        elif color_by == 'games':
            node_colors = [self.state_graph.nodes[node]['num_games'] 
                          for node in self.state_graph.nodes()]
            cmap = 'plasma'
            label = 'Games Visited'
        elif color_by == 'depth':
            node_colors = [self.state_metadata[node]['depth']
                          for node in self.state_graph.nodes()]
            cmap = 'cool'
            label = 'Depth (Move Number)'
        else:
            node_colors = 'skyblue'
            cmap = None
            label = None
        
        edges = self.state_graph.edges()
        
        # Simplified edges - just simple lines, no thickness variation
        nx.draw_networkx_edges(
            self.state_graph, pos,
            width=1.0,
            alpha=0.4,
            edge_color='gray',
            arrows=True,
            arrowsize=10,
            arrowstyle='->',
            ax=ax
        )
        
        nx.draw_networkx_nodes(
            self.state_graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            alpha=0.8,
            ax=ax
        )
        
        if show_labels:
            labels = {node: f"{self.state_graph.nodes[node]['visits']}" 
                     for node in self.state_graph.nodes()}
            nx.draw_networkx_labels(self.state_graph, pos, labels, 
                                   font_size=8, ax=ax)
        
        # No colorbar/legend
        ax.set_title(f'Game State Exploration Tree\n'
                    f'{self.state_graph.number_of_nodes()} unique states, '
                    f'{self.state_graph.number_of_edges()} transitions, '
                    f'{len(self.games_data)} games',
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def print_statistics(self):
        """Print detailed statistics about state exploration"""
        print("\n" + "="*70)
        print("STATE EXPLORATION STATISTICS")
        print("="*70)
        
        print(f"\nGames analyzed: {len(self.games_data)}")
        print(f"Unique states discovered: {len(self.state_visits)}")
        print(f"Total state visits: {sum(self.state_visits.values())}")
        print(f"Unique transitions: {len(self.transition_counts)}")
        print(f"Total transitions: {sum(self.transition_counts.values())}")
        
        if self.state_visits:
            avg_visits = sum(self.state_visits.values()) / len(self.state_visits)
            print(f"\nAverage visits per state: {avg_visits:.2f}")
            
            most_visited = self.state_visits.most_common(10)
            print("\nTop 10 most visited states:")
            for i, (state_hash, count) in enumerate(most_visited, 1):
                metadata = self.state_metadata[state_hash]
                print(f"  {i:2d}. State {state_hash[:8]}... - "
                      f"{count} visits, {len(metadata['games_visited'])} games, "
                      f"turn {metadata['turn']}")
        
        if self.transition_counts:
            most_common_transitions = sorted(self.transition_counts.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]
            print("\nTop 10 most common transitions:")
            for i, ((from_state, to_state), count) in enumerate(most_common_transitions, 1):
                print(f"  {i:2d}. {from_state[:8]}... -> {to_state[:8]}... "
                      f"({count} times)")
        
        if self.state_graph.number_of_nodes() > 0:
            avg_degree = sum(dict(self.state_graph.degree()).values()) / self.state_graph.number_of_nodes()
            print(f"\nGraph connectivity:")
            print(f"  Average node degree: {avg_degree:.2f}")
            print(f"  Graph density: {nx.density(self.state_graph):.4f}")
        
        print("\n" + "="*70)
    
    def get_state_details(self, state_hash: str) -> Dict:
        """
        Get detailed information about a specific state
        
        Args:
            state_hash (str): Hash of the state to query
            
        Returns:
            dict: State details including board, visits, transitions
        """
        if state_hash not in self.hash_to_state:
            return None
        
        return {
            'hash': state_hash,
            'board': self.hash_to_state[state_hash],
            'visits': self.state_visits[state_hash],
            'metadata': self.state_metadata[state_hash],
            'incoming_transitions': list(self.state_graph.predecessors(state_hash)) 
                                   if state_hash in self.state_graph else [],
            'outgoing_transitions': list(self.state_graph.successors(state_hash))
                                   if state_hash in self.state_graph else []
        }
