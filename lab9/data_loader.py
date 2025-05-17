import os
import numpy as np
import json
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix

def load_email_network_data(file_path='./email-Eu-core-temporal/email-Eu-core-temporal.txt'):
    """
    Load the email-Eu-core-temporal network data and construct a weighted adjacency matrix.
    
    Args:
        file_path: Path to the .txt file containing the email network data.
                  Format expected: SRC DST UNIXTS (space-separated)
    
    Returns:
        tuple: (W_net, n_users)
            - W_net (np.ndarray): Weighted adjacency matrix where weights are proportional
                                 to interaction frequency, row-normalized.
            - n_users (int): Number of unique users in the network.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        # Try path relative to script location if absolute fails
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Email network data not found at {file_path}")
    
    # Load the data
    interactions = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    src = int(parts[0])
                    dst = int(parts[1])
                    # timestamp = int(parts[2])  # We don't need timestamp for frequency calculation
                    interactions.append((src, dst))
                else:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
    except Exception as e:
        raise IOError(f"Error reading email network data: {e}")
    
    if not interactions:
        raise ValueError("No valid interactions found in the data file")
    
    # Find all unique node IDs and create mapping to 0...n-1 (for continuous indexing)
    all_nodes = set()
    for src, dst in interactions:
        all_nodes.add(src)
        all_nodes.add(dst)
    
    node_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    n_users = len(node_map)
    
    # Count interaction frequencies
    interaction_counts = {}
    for src, dst in interactions:
        src_idx = node_map[src]
        dst_idx = node_map[dst]
        if src_idx not in interaction_counts:
            interaction_counts[src_idx] = {}
        if dst_idx not in interaction_counts[src_idx]:
            interaction_counts[src_idx][dst_idx] = 0
        interaction_counts[src_idx][dst_idx] += 1
    
    # Create weighted adjacency matrix
    W_net = np.zeros((n_users, n_users))
    for src_idx in range(n_users):
        if src_idx in interaction_counts:
            total_outgoing = sum(interaction_counts[src_idx].values())
            for dst_idx, count in interaction_counts[src_idx].items():
                # Normalize by row (sender's total interactions)
                W_net[src_idx, dst_idx] = count / total_outgoing if total_outgoing > 0 else 0
    
    return W_net, n_users


def load_wiki_vote_data(file_path='./wiki-vote/wiki-Vote.txt'):
    """
    Load the wiki-Vote network data and construct a weighted adjacency matrix.
    
    Args:
        file_path: Path to the .txt file containing the wiki-Vote data.
                  Format expected: SRC DST (space-separated)
    
    Returns:
        tuple: (W_net, n_users)
            - W_net (np.ndarray): Weighted adjacency matrix where weights are proportional
                                 to interaction frequency, row-normalized.
            - n_users (int): Number of unique users in the network.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        # Try path relative to script location if absolute fails
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Wiki-Vote network data not found at {file_path}")
    
    # Load the data
    interactions = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Skip comment lines
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split()
                if len(parts) >= 2:
                    src = int(parts[0])
                    dst = int(parts[1])
                    interactions.append((src, dst))
                else:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
    except Exception as e:
        raise IOError(f"Error reading Wiki-Vote network data: {e}")
    
    if not interactions:
        raise ValueError("No valid interactions found in the data file")
    
    # Find all unique node IDs and create mapping to 0...n-1 (for continuous indexing)
    all_nodes = set()
    for src, dst in interactions:
        all_nodes.add(src)
        all_nodes.add(dst)
    
    node_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    n_users = len(node_map)
    
    # Count interaction frequencies
    interaction_counts = {}
    for src, dst in interactions:
        src_idx = node_map[src]
        dst_idx = node_map[dst]
        if src_idx not in interaction_counts:
            interaction_counts[src_idx] = {}
        if dst_idx not in interaction_counts[src_idx]:
            interaction_counts[src_idx][dst_idx] = 0
        interaction_counts[src_idx][dst_idx] += 1
    
    # Create weighted adjacency matrix
    W_net = np.zeros((n_users, n_users))
    for src_idx in range(n_users):
        if src_idx in interaction_counts:
            total_outgoing = sum(interaction_counts[src_idx].values())
            for dst_idx, count in interaction_counts[src_idx].items():
                # Normalize by row (voter's total votes)
                W_net[src_idx, dst_idx] = count / total_outgoing if total_outgoing > 0 else 0
    
    return W_net, n_users


def load_facebook_data(file_path='./ego-fb/facebook_combined.txt'):
    """
    Load the Facebook combined network data and construct a weighted adjacency matrix.
    
    Args:
        file_path: Path to the .txt file containing the Facebook network data.
                  Format expected: SRC DST (space-separated)
    
    Returns:
        tuple: (W_net, n_users)
            - W_net (np.ndarray): Weighted adjacency matrix where weights are proportional
                                 to friendship connections, row-normalized.
            - n_users (int): Number of unique users in the network.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        # Try path relative to script location if absolute fails
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Facebook network data not found at {file_path}")
    
    # Load the data
    interactions = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Skip comment lines
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split()
                if len(parts) >= 2:
                    src = int(parts[0])
                    dst = int(parts[1])
                    interactions.append((src, dst))
                    # Facebook network is undirected, add the reverse edge too
                    interactions.append((dst, src))
                else:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
    except Exception as e:
        raise IOError(f"Error reading Facebook network data: {e}")
    
    if not interactions:
        raise ValueError("No valid interactions found in the data file")
    
    # Find all unique node IDs and create mapping to 0...n-1 (for continuous indexing)
    all_nodes = set()
    for src, dst in interactions:
        all_nodes.add(src)
        all_nodes.add(dst)
    
    node_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    n_users = len(node_map)
    
    # Count interaction frequencies
    interaction_counts = {}
    for src, dst in interactions:
        src_idx = node_map[src]
        dst_idx = node_map[dst]
        if src_idx not in interaction_counts:
            interaction_counts[src_idx] = {}
        if dst_idx not in interaction_counts[src_idx]:
            interaction_counts[src_idx][dst_idx] = 0
        interaction_counts[src_idx][dst_idx] += 1
    
    # Create weighted adjacency matrix
    W_net = np.zeros((n_users, n_users))
    for src_idx in range(n_users):
        if src_idx in interaction_counts:
            total_outgoing = sum(interaction_counts[src_idx].values())
            for dst_idx, count in interaction_counts[src_idx].items():
                # Normalize by row (user's total connections)
                W_net[src_idx, dst_idx] = count / total_outgoing if total_outgoing > 0 else 0
    
    return W_net, n_users


def load_lastfm_asia_data(file_path='./last-fm/lasftm_asia/lastfm_asia_edges.csv'):
    """
    Load the Last.fm Asia network data and construct a weighted adjacency matrix.
    
    Args:
        file_path: Path to the CSV file containing the Last.fm Asia network data.
                  Format expected: SRC,DST (comma-separated) with a possible header row
    
    Returns:
        tuple: (W_net, n_users)
            - W_net (np.ndarray): Weighted adjacency matrix where weights are proportional
                                 to friendship connections, row-normalized.
            - n_users (int): Number of unique users in the network.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        # Try path relative to script location if absolute fails
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Last.fm Asia network data not found at {file_path}")
    
    # Load the data
    interactions = []
    try:
        with open(file_path, 'r') as f:
            # Skip the first line which is likely the header
            first_line = f.readline().strip()
            if first_line and ('node' in first_line.lower() or 'source' in first_line.lower() or 
                              'from' in first_line.lower() or 'target' in first_line.lower() or 
                              'to' in first_line.lower() or 'dst' in first_line.lower() or
                              'src' in first_line.lower()):
                # This was a header, continue with the rest of the file
                pass
            else:
                # If it wasn't a header, process it as data
                parts = first_line.split(',')
                if len(parts) >= 2:
                    try:
                        src = int(parts[0])
                        dst = int(parts[1])
                        interactions.append((src, dst))
                        # Last.fm network is undirected, add the reverse edge too
                        interactions.append((dst, src))
                    except ValueError:
                        # If we can't convert to integers, this was actually a header
                        print(f"Detected header: {first_line}")
                else:
                    print(f"Warning: Skipping malformed line: {first_line}")
            
            # Process the rest of the file
            for line in f:
                # Skip comment lines
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        src = int(parts[0])
                        dst = int(parts[1])
                        interactions.append((src, dst))
                        # Last.fm network is undirected, add the reverse edge too
                        interactions.append((dst, src))
                    except ValueError as e:
                        print(f"Warning: Skipping line with non-integer IDs: {line.strip()}")
                else:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
    except Exception as e:
        raise IOError(f"Error reading Last.fm Asia network data: {e}")
    
    if not interactions:
        raise ValueError("No valid interactions found in the data file")
    
    # Find all unique node IDs and create mapping to 0...n-1 (for continuous indexing)
    all_nodes = set()
    for src, dst in interactions:
        all_nodes.add(src)
        all_nodes.add(dst)
    
    node_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    n_users = len(node_map)
    
    # Count interaction frequencies
    interaction_counts = {}
    for src, dst in interactions:
        src_idx = node_map[src]
        dst_idx = node_map[dst]
        if src_idx not in interaction_counts:
            interaction_counts[src_idx] = {}
        if dst_idx not in interaction_counts[src_idx]:
            interaction_counts[src_idx][dst_idx] = 0
        interaction_counts[src_idx][dst_idx] += 1
    
    # Create weighted adjacency matrix
    W_net = np.zeros((n_users, n_users))
    for src_idx in range(n_users):
        if src_idx in interaction_counts:
            total_outgoing = sum(interaction_counts[src_idx].values())
            for dst_idx, count in interaction_counts[src_idx].items():
                # Normalize by row (user's total connections)
                W_net[src_idx, dst_idx] = count / total_outgoing if total_outgoing > 0 else 0
    
    return W_net, n_users


def load_bitcoin_data(file_path='./soc-bitcoin/soc-sign-bitcoinalpha.csv'):
    """
    Load the Bitcoin Alpha network data and construct a weighted adjacency matrix.
    
    The network represents who-trusts-whom relationships where users rate others
    on a scale from -10 to +10. We use these ratings as edge weights.
    
    Args:
        file_path: Path to the CSV file containing the Bitcoin network data.
                  Format expected: SOURCE,TARGET,RATING,TIME (comma-separated)
    
    Returns:
        tuple: (W_net, n_users)
            - W_net (np.ndarray): Weighted adjacency matrix where weights are based on ratings,
                                 normalized to [0,1] range and row-normalized.
            - n_users (int): Number of unique users in the network.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        # Try path relative to script location if absolute fails
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Bitcoin network data not found at {file_path}")
    
    # Load the data
    interactions = []
    try:
        with open(file_path, 'r') as f:
            # Skip the header if present
            first_line = f.readline().strip()
            if first_line and ('source' in first_line.lower() or 'target' in first_line.lower() or 
                              'rating' in first_line.lower() or 'time' in first_line.lower()):
                # This was a header, continue with the rest of the file
                print(f"Detected header: {first_line}")
            else:
                # If it wasn't a header, process it as data
                parts = first_line.split(',')
                if len(parts) >= 3:  # We need at least source, target, and rating
                    try:
                        src = int(parts[0])
                        dst = int(parts[1])
                        rating = float(parts[2])
                        interactions.append((src, dst, rating))
                    except ValueError:
                        # If we can't convert to integers/float, this was actually a header
                        print(f"Detected non-numeric data (likely header): {first_line}")
                else:
                    print(f"Warning: Skipping malformed line: {first_line}")
            
            # Process the rest of the file
            for line in f:
                # Skip comment lines
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split(',')
                if len(parts) >= 3:  # We need at least source, target, and rating
                    try:
                        src = int(parts[0])
                        dst = int(parts[1])
                        rating = float(parts[2])
                        interactions.append((src, dst, rating))
                    except ValueError as e:
                        print(f"Warning: Skipping line with non-numeric data: {line.strip()}")
                else:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
    except Exception as e:
        raise IOError(f"Error reading Bitcoin network data: {e}")
    
    if not interactions:
        raise ValueError("No valid interactions found in the data file")
    
    # Find all unique node IDs and create mapping to 0...n-1 (for continuous indexing)
    all_nodes = set()
    for src, dst, _ in interactions:
        all_nodes.add(src)
        all_nodes.add(dst)
    
    node_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    n_users = len(node_map)
    
    # Rating min/max for normalization
    min_rating = min(rating for _, _, rating in interactions)
    max_rating = max(rating for _, _, rating in interactions)
    rating_range = max_rating - min_rating
    
    # Initialize with zeros for all unknown ratings
    raw_weights = np.zeros((n_users, n_users))
    
    # Sum of ratings (can be positive and negative)
    for src, dst, rating in interactions:
        src_idx = node_map[src]
        dst_idx = node_map[dst]
        # Normalize rating to [0,1] range
        # From the readme, ratings are from -10 to +10, so we add 10 and divide by 20
        normalized_rating = (rating - min_rating) / rating_range if rating_range > 0 else 0.5
        raw_weights[src_idx, dst_idx] = normalized_rating
    
    # Row-normalize the adjacency matrix
    W_net = np.zeros((n_users, n_users))
    for i in range(n_users):
        row_sum = np.sum(raw_weights[i, :])
        if row_sum > 0:
            W_net[i, :] = raw_weights[i, :] / row_sum
    
    return W_net, n_users


def load_college_msg_data(file_path='./CollegeMsg_temp/CollegeMsg.txt'):
    """
    Load the CollegeMsg temporal network data and construct a weighted adjacency matrix.
    
    The network represents private messages sent between users at UC Irvine.
    Edge weights are proportional to the frequency of interactions.
    
    Args:
        file_path: Path to the .txt file containing the CollegeMsg data.
                  Format expected: SRC DST UNIXTS (space-separated)
    
    Returns:
        tuple: (W_net, n_users)
            - W_net (np.ndarray): Weighted adjacency matrix where weights are proportional
                                 to message frequency, row-normalized.
            - n_users (int): Number of unique users in the network.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        # Try path relative to script location if absolute fails
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CollegeMsg network data not found at {file_path}")
    
    # Load the data
    interactions = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Skip comment lines
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split()
                if len(parts) >= 3:
                    src = int(parts[0])
                    dst = int(parts[1])
                    # timestamp = int(parts[2])  # We don't need timestamp for frequency calculation
                    interactions.append((src, dst))
                else:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
    except Exception as e:
        raise IOError(f"Error reading CollegeMsg network data: {e}")
    
    if not interactions:
        raise ValueError("No valid interactions found in the data file")
    
    # Find all unique node IDs and create mapping to 0...n-1 (for continuous indexing)
    all_nodes = set()
    for src, dst in interactions:
        all_nodes.add(src)
        all_nodes.add(dst)
    
    node_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    n_users = len(node_map)
    
    # Count interaction frequencies
    interaction_counts = {}
    for src, dst in interactions:
        src_idx = node_map[src]
        dst_idx = node_map[dst]
        if src_idx not in interaction_counts:
            interaction_counts[src_idx] = {}
        if dst_idx not in interaction_counts[src_idx]:
            interaction_counts[src_idx][dst_idx] = 0
        interaction_counts[src_idx][dst_idx] += 1
    
    # Create weighted adjacency matrix
    W_net = np.zeros((n_users, n_users))
    for src_idx in range(n_users):
        if src_idx in interaction_counts:
            total_outgoing = sum(interaction_counts[src_idx].values())
            for dst_idx, count in interaction_counts[src_idx].items():
                # Normalize by row (sender's total messages)
                W_net[src_idx, dst_idx] = count / total_outgoing if total_outgoing > 0 else 0
    
    return W_net, n_users


def load_mooc_data(file_path='./mooc/act-mooc/mooc_actions.tsv'):
    """
    Load the MOOC temporal network data and construct a weighted adjacency matrix.
    
    The network represents actions taken by users on a MOOC platform.
    Each edge represents a user (source) taking an action on a target activity.
    Edge weights are proportional to the frequency of interactions.
    
    Args:
        file_path: Path to the TSV file containing the MOOC data.
                  Format expected: ACTIONID USERID TARGETID TIMESTAMP (tab-separated)
    
    Returns:
        tuple: (W_net, n_users)
            - W_net (np.ndarray): Weighted adjacency matrix where weights are proportional
                                 to interaction frequency, row-normalized.
            - n_users (int): Number of unique users in the network.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        # Try path relative to script location if absolute fails
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MOOC network data not found at {file_path}")
    
    # Load the data
    interactions = []
    try:
        with open(file_path, 'r') as f:
            # Skip the header line
            header = f.readline()
            if not ('USERID' in header and 'TARGETID' in header):
                print(f"Warning: Expected header with USERID and TARGETID, but found: {header.strip()}")
            
            # Process the rest of the file
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:  # ACTIONID, USERID, TARGETID, TIMESTAMP
                    try:
                        # We can ignore ACTIONID as instructed
                        # action_id = int(parts[0])
                        user_id = int(parts[1])
                        target_id = int(parts[2])
                        # timestamp = float(parts[3])  # Not needed for frequency calculation
                        interactions.append((user_id, target_id))
                    except ValueError as e:
                        print(f"Warning: Skipping line with non-integer IDs: {line.strip()}")
                else:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
    except Exception as e:
        raise IOError(f"Error reading MOOC network data: {e}")
    
    if not interactions:
        raise ValueError("No valid interactions found in the data file")
    
    # Find all unique user and target IDs for node mapping
    users = set()
    targets = set()
    for user_id, target_id in interactions:
        users.add(user_id)
        targets.add(target_id)
    
    # Create a mapping for all nodes (users and targets)
    all_nodes = sorted(list(users.union(targets)))
    node_map = {node: idx for idx, node in enumerate(all_nodes)}
    n_total_nodes = len(node_map)
    n_users = len(users)
    
    # Count interaction frequencies
    interaction_counts = {}
    for user_id, target_id in interactions:
        user_idx = node_map[user_id]
        target_idx = node_map[target_id]
        if user_idx not in interaction_counts:
            interaction_counts[user_idx] = {}
        if target_idx not in interaction_counts[user_idx]:
            interaction_counts[user_idx][target_idx] = 0
        interaction_counts[user_idx][target_idx] += 1
    
    # Create weighted adjacency matrix
    W_net = np.zeros((n_total_nodes, n_total_nodes))
    for src_idx in range(n_total_nodes):
        if src_idx in interaction_counts:
            total_outgoing = sum(interaction_counts[src_idx].values())
            for dst_idx, count in interaction_counts[src_idx].items():
                # Normalize by row (user's total actions)
                W_net[src_idx, dst_idx] = count / total_outgoing if total_outgoing > 0 else 0
    
    return W_net, n_total_nodes


def load_deezer_ego_nets_data(file_path="data/deezer_ego_nets/deezer_edges.json"):
    """
    Loads the Deezer ego nets dataset, selects the largest graph by node count,
    and constructs its weighted adjacency matrix.
    Edges are unweighted and undirected.
    """
    with open(file_path, 'r') as f:
        all_graphs_edges = json.load(f) # Assuming the JSON is a list of edge lists or dict of edge lists

    largest_graph_edges = []
    max_nodes = -1
    
    # The structure of all_graphs_edges needs to be determined.
    # Common formats:
    # 1. A dictionary: {graph_id: [[node1, node2], [node2, node3], ...], ...}
    # 2. A list of graphs, where each graph is an edge list: [[[node1, node2], ...], [[node_a, node_b], ...]]
    # For now, let's assume it's a dictionary of graph_id to edge list.
    # If it's a list of graphs, the iteration logic will be similar.
    
    # We need to iterate through whatever structure `all_graphs_edges` is.
    # Let's assume it's a dictionary where keys are graph IDs and values are the edge lists.
    # If it's a list of graphs, where each graph is itself a list of edges, the logic would be:
    # for graph_edges in all_graphs_edges:
    #   current_nodes = set()
    #   for edge in graph_edges:
    #       current_nodes.add(edge[0])
    #       current_nodes.add(edge[1])
    #   if len(current_nodes) > max_nodes:
    #       max_nodes = len(current_nodes)
    #       largest_graph_edges = graph_edges

    # Based on common JSON structures for multiple graphs, it's often a dictionary
    # where keys are graph IDs (e.g., strings) and values are the list of edges for that graph.
    
    processed_graphs = 0
    if isinstance(all_graphs_edges, dict):
        for graph_id in all_graphs_edges:
            current_graph_edges = all_graphs_edges[graph_id]
            nodes_in_current_graph = set()
            for edge in current_graph_edges:
                nodes_in_current_graph.add(edge[0])
                nodes_in_current_graph.add(edge[1])
            
            if len(nodes_in_current_graph) > max_nodes:
                max_nodes = len(nodes_in_current_graph)
                largest_graph_edges = current_graph_edges
            processed_graphs +=1
            
    elif isinstance(all_graphs_edges, list): # Assuming a list of edge lists
        for current_graph_edges in all_graphs_edges:
            nodes_in_current_graph = set()
            if isinstance(current_graph_edges, list): # each item is an edge list
                 for edge in current_graph_edges:
                    nodes_in_current_graph.add(edge[0])
                    nodes_in_current_graph.add(edge[1])
            # This part seems less likely for a large number of graphs, but covering bases.
            # Or it could be a list of graph objects, each having an 'edges' key and 'nodes' key or similar.
            # Without knowing the exact structure, this is a guess.
            # The prompt mentioned "deezer_edges.json file contains a collection of undirected graphs"
            # Let's assume it's a list of lists of edges.
            # Example: [ [[1,2],[2,3]], [[10,11],[11,12]] ] -> two graphs

            if len(nodes_in_current_graph) > max_nodes:
                max_nodes = len(nodes_in_current_graph)
                largest_graph_edges = current_graph_edges
            processed_graphs +=1
    else:
        # Fallback or error if the structure is unexpected
        print(f"Unexpected JSON structure: {type(all_graphs_edges)}")
        return None, 0

    if not largest_graph_edges:
        print("No graph data found or processed.")
        return None, 0
        
    # print(f"Found {processed_graphs} graphs. Largest graph has {max_nodes} nodes.")
    # Expected max_nodes is 363 according to README

    unique_nodes = set()
    for edge in largest_graph_edges:
        unique_nodes.add(edge[0])
        unique_nodes.add(edge[1])

    node_to_idx = {node: i for i, node in enumerate(sorted(list(unique_nodes)))}
    n_users = len(unique_nodes)
    
    W_net = np.zeros((n_users, n_users))

    for edge in largest_graph_edges:
        u, v = edge
        if u in node_to_idx and v in node_to_idx:
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            # Undirected and unweighted graph, so weight is 1
            W_net[u_idx, v_idx] = 1
            W_net[v_idx, u_idx] = 1 
            # If there were frequencies/multiple edges, we'd increment here.
            # But README says "Edge features: No", implying simple existence.

    return W_net, n_users


def load_github_stargazers_data(file_path="data/github_stargazers/git_edges.json"):
    """
    Loads the GitHub Stargazers dataset.
    The dataset contains a collection of graphs where nodes are users who starred GitHub repositories.
    Each graph represents follower relationships between users who starred similar repositories.
    
    According to README.txt:
    - Directed: No (undirected graphs)
    - Node features: No
    - Edge features: No
    - Binary-labeled: Yes (whether developers are web or ML focused)
    
    This function:
    1. Loads the largest graph from the collection
    2. Creates a user-user adjacency matrix
    3. Row-normalizes the matrix
    
    Args:
        file_path: Path to the JSON file containing the edge list
        
    Returns:
        tuple: (W_net, n_users)
            - W_net: Row-normalized adjacency matrix (dense numpy array)
            - n_users: Number of unique users in the graph
    """
    print(f"Loading GitHub Stargazers network data from {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        # Based on README.txt, the dataset contains multiple graphs
        # The JSON structure is likely to be either:
        # 1. A list of graphs, each with its own edge list: [graph1, graph2, ...] where graph1 = [[user1, user2], [user2, user3], ...]
        # 2. A dict mapping graph IDs to edge lists: {"graph1": [[user1, user2], ...], "graph2": [[user3, user4], ...]}
        
        # First, determine the structure and extract the largest graph
        largest_graph = None
        largest_graph_size = 0
        
        if isinstance(json_data, list):
            # Case 1: List of graphs or list of edges
            # Check if it's a list of edges directly (single graph) or list of graphs
            if json_data and isinstance(json_data[0], list):
                sample_item = json_data[0]
                if len(sample_item) == 2 and all(isinstance(x, (int, str)) for x in sample_item):
                    # This is likely a direct edge list [user1, user2]
                    print("Detected direct edge list format.")
                    largest_graph = json_data
                    largest_graph_size = len(set(item for sublist in json_data for item in sublist))
                else:
                    # This is likely a list of graphs, each with its own edge list
                    print("Detected list of graphs format.")
                    for graph in json_data:
                        # Assume each graph is a list of [user1, user2] edges
                        if graph and isinstance(graph, list):
                            # Count unique nodes in this graph
                            nodes = set()
                            for edge in graph:
                                if isinstance(edge, list) and len(edge) == 2:  
                                    nodes.add(edge[0])
                                    nodes.add(edge[1])
                            
                            graph_size = len(nodes)
                            if graph_size > largest_graph_size:
                                largest_graph_size = graph_size
                                largest_graph = graph
            else:
                print("Unexpected JSON structure in list format.")
        
        elif isinstance(json_data, dict):
            # Case 2: Dictionary mapping graph IDs to edge lists
            print("Detected dictionary of graphs format.")
            for graph_id, edges in json_data.items():
                if edges and isinstance(edges, list):
                    # Count unique nodes in this graph
                    nodes = set()
                    for edge in edges:
                        if isinstance(edge, list) and len(edge) == 2:
                            nodes.add(edge[0])
                            nodes.add(edge[1])
                    
                    graph_size = len(nodes)
                    if graph_size > largest_graph_size:
                        largest_graph_size = graph_size
                        largest_graph = edges
        
        else:
            print(f"Unexpected JSON structure: {type(json_data)}")
            return np.zeros((0, 0)), 0

        if largest_graph is None or largest_graph_size == 0:
            print("No valid graph found in the JSON data.")
            return np.zeros((0, 0)), 0
            
        print(f"Found largest graph with {largest_graph_size} nodes.")
        
        # First validate all edges and collect valid nodes
        valid_edges = []
        unique_nodes = set()
        
        for edge in largest_graph:
            # Check if edge is properly formatted as a 2-element list/tuple
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                print(f"Skipping malformed edge: {edge}")
                continue
                
            u, v = edge
            valid_edges.append((u, v))
            unique_nodes.add(u)
            unique_nodes.add(v)
        
        # Map nodes to consecutive indices
        node_to_idx = {node: i for i, node in enumerate(sorted(list(unique_nodes)))}
        n_users = len(unique_nodes)
        
        if n_users == 0:
            print("No valid nodes found in the largest graph.")
            return np.zeros((0, 0)), 0
            
        # Create adjacency matrix as dense numpy array (like in load_deezer_ego_nets_data)
        W_net = np.zeros((n_users, n_users))
        
        for u, v in valid_edges:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            # Undirected graph, so add edges in both directions
            W_net[u_idx, v_idx] = 1.0
            W_net[v_idx, u_idx] = 1.0
        
        # Row-normalize the matrix
        row_sums = W_net.sum(axis=1)
        for i in range(n_users):
            if row_sums[i] > 0:
                W_net[i, :] = W_net[i, :] / row_sums[i]
        
        print(f"GitHub Stargazers data loaded: {n_users} users, {int(np.sum(W_net > 0))} edges.")
        return W_net, n_users
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Ensure it's a valid JSON file.")
        raise
    except Exception as e:
        print(f"Error reading GitHub Stargazers network data: {e}")
        raise


def get_all_network_statistics():
    """
    Precomputes statistics for all network datasets to facilitate setting appropriate 
    multiple_n values in experiments.
    
    Returns:
        dict: Dictionary mapping dataset names to their statistics:
            - 'n_users': Number of nodes/users in the network
            - 'n_edges': Number of edges in the network
            - 'density': Edge density (n_edges / possible_edges)
            - 'recommended_multiple_n': Recommended multiple_n value (n_users/100.0)
    """
    network_stats = {}
    
    # Dictionary mapping dataset names to their loader functions and default file paths
    datasets = {
        'email': (load_email_network_data, './email-Eu-core-temporal/email-Eu-core-temporal.txt'),
        'wiki_vote': (load_wiki_vote_data, './wiki-vote/wiki-Vote.txt'),
        'facebook': (load_facebook_data, './ego-fb/facebook_combined.txt'),
        'lastfm_asia': (load_lastfm_asia_data, './last-fm/lasftm_asia/lastfm_asia_edges.csv'),
        'bitcoin': (load_bitcoin_data, './soc-bitcoin/soc-sign-bitcoinalpha.csv'),
        'college_msg': (load_college_msg_data, './CollegeMsg_temp/CollegeMsg.txt'),
        'mooc': (load_mooc_data, './mooc/act-mooc/mooc_actions.tsv'),
        'deezer_ego_nets': (load_deezer_ego_nets_data, './deezer_ego_nets/deezer_edges.json'),
        'github_stargazers': (load_github_stargazers_data, './github_stargazers/git_edges.json')
    }
    
    print("Computing network statistics for all datasets...")
    
    for dataset_name, (loader_func, default_path) in datasets.items():
        try:
            print(f"Loading {dataset_name} dataset...")
            W_net, n_users = loader_func(default_path)
            
            # Calculate number of edges (non-zero entries in W_net)
            n_edges = int(np.sum(W_net > 0))
            
            # Calculate density
            possible_edges = n_users * (n_users - 1)  # Directed graph
            density = n_edges / possible_edges if possible_edges > 0 else 0
            
            # Calculate recommended multiple_n (n_users/100.0)
            recommended_multiple_n = n_users / 100.0
            
            # Store statistics
            network_stats[dataset_name] = {
                'n_users': n_users,
                'n_edges': n_edges,
                'density': density,
                'recommended_multiple_n': recommended_multiple_n
            }
            
            print(f"  {dataset_name}: {n_users} nodes, {n_edges} edges, density={density:.6f}, recommended_multiple_n={recommended_multiple_n:.2f}")
            
        except Exception as e:
            print(f"Error computing statistics for {dataset_name}: {e}")
            network_stats[dataset_name] = {
                'error': str(e)
            }
    
    # Also handle Congress network data which is loaded differently
    try:
        print("Loading Congress network data...")
        import json
        import os
        
        json_path = './congress_network/congress_network_data.json'
        if not os.path.exists(json_path):
            script_dir = os.path.dirname(__file__)
            json_path = os.path.join(script_dir, json_path)
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list): 
            data = data[0]
            
        username_list = data['usernameList']
        n_users = len(username_list)
        
        # Count edges (this might take some time for large networks)
        n_edges = 0
        for outlist in data['outList']:
            n_edges += len(outlist)
        
        possible_edges = n_users * (n_users - 1)
        density = n_edges / possible_edges if possible_edges > 0 else 0
        recommended_multiple_n = n_users / 100.0
        
        network_stats['congress'] = {
            'n_users': n_users,
            'n_edges': n_edges,
            'density': density,
            'recommended_multiple_n': recommended_multiple_n
        }
        
        print(f"  congress: {n_users} nodes, {n_edges} edges, density={density:.6f}, recommended_multiple_n={recommended_multiple_n:.2f}")
        
    except Exception as e:
        print(f"Error computing statistics for Congress network: {e}")
        network_stats['congress'] = {
            'error': str(e)
        }
    
    return network_stats

def save_network_statistics(output_file='network_statistics.json'):
    """
    Computes and saves network statistics to a JSON file.
    
    Args:
        output_file: Path to save the JSON file
    
    Returns:
        dict: The computed network statistics
    """
    stats = get_all_network_statistics()
    
    try:
        import json
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Network statistics saved to {output_file}")
    except Exception as e:
        print(f"Error saving network statistics: {e}")
    
    return stats

if __name__ == "__main__":
    # If this script is run directly, compute and save network statistics
    save_network_statistics()

