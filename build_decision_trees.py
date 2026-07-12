# I use this file to build a complete decision tree..

#Yes, we can extend the code to handle a dataset with any number of attributes 
#(categorical or numeric) and a binary outcome label ("Yes" or "No"). The generalized version 
#loads data from a CSV file (assuming the last column is the label), computes root entropy, 
#and calculates information gain for each attribute:
#For categorical attributes: Splits on unique values and computes IG.
#For numeric attributes: Finds the best threshold by testing midpoints between sorted unique
#values

# Decision Tree Entropy & Information Gain Computation
# Generalized for datasets with n attributes and Yes/No outcome labels
# Extended to build full decision tree with recursive branch node calculations
# Added Graphviz visualization

import math
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from graphviz import Digraph
import html

# -----------------------------
# Calculate Entropy and Information Gain
def entropy(labels):
    """Compute entropy of a list of class labels"""
    total = len(labels)
    if total == 0:
        return 0.0
    counts = Counter(labels)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def information_gain(parent_labels, subsets):
    """Compute information gain given parent labels and subsets"""
    parent_entropy = entropy(parent_labels)
    total = len(parent_labels)
    if total == 0:
        return 0.0
    weighted_entropy = 0.0
    for subset in subsets:
        weighted_entropy += (len(subset) / total) * entropy(subset)
    return parent_entropy - weighted_entropy

#------------------------------
def safe_text(x):
    """Robust sanitizer for XES + CSV mixed data"""
    if x is None:
        return "UNKNOWN"

    x = str(x).strip()

    if x == "" or x.lower() in ["nan", "none"]:
        return "UNKNOWN"

    return html.escape(x)
# -----------------------------
# Decision Tree Node Class
# -----------------------------

class Node:
    _node_counter = 0  # Class variable for unique node IDs
    
    def __init__(self, data, attributes, depth=0, max_depth=10, min_samples=5):
        self.attribute = None
        self.threshold = None
        self.children = {}
        self.is_leaf = False
        self.label = None
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.node_id = Node._node_counter  # Unique ID for graphviz
        Node._node_counter += 1

        labels = data.iloc[:, -1].tolist()
        if not labels:
            self.is_leaf = True
            self.label = None
            return
        self.entropy = entropy(labels)
        self.samples = len(labels)

        # Check stopping conditions
        if depth >= max_depth or len(attributes) == 0 or len(set(labels)) == 1 or self.samples < min_samples:
            self.is_leaf = True
            self.label = Counter(labels).most_common(1)[0][0]
            return

        # Find best split
        best_ig = -1
        best_attr = None
        best_thresh = None
        best_subsets = None

        for attr in attributes:
            if data[attr].dtype == 'object':  # Categorical
                unique_vals = data[attr].unique()
                subsets = [data[data[attr] == val][data.columns[-1]].tolist() for val in unique_vals]
                if any(len(s) == 0 for s in subsets):
                    continue
                ig = information_gain(labels, subsets)
                if ig > best_ig:
                    best_ig = ig
                    best_attr = attr
                    best_subsets = subsets
                    best_thresh = None
            else:  # Numeric
                #sorted_vals = sorted(data[attr].unique()) # It creates parsing issue on Adrian's machine, so I will use the following instead
                # Use pd.to_numeric to ensure numeric conversion and handle non-numeric values
                #errors="coerce" will convert non-numeric values to NaN, which will be dropped in the next step
                #dropna() will remove NaN values, ensuring that we only consider valid numeric values for threshold calculation
                sorted_vals = sorted(
                   pd.to_numeric(data[attr], errors="coerce").dropna().unique()
                )
                if len(sorted_vals) <= 1:
                    continue
                for i in range(len(sorted_vals) - 1):
                    thresh = (sorted_vals[i] + sorted_vals[i + 1]) / 2
                    left = data[data[attr] <= thresh][data.columns[-1]].tolist()
                    right = data[data[attr] > thresh][data.columns[-1]].tolist()
                    if len(left) == 0 or len(right) == 0:
                        continue
                    ig = information_gain(labels, [left, right])
                    if ig > best_ig:
                        best_ig = ig
                        best_attr = attr
                        best_thresh = thresh
                        best_subsets = [left, right]

        if best_attr is None:
            self.is_leaf = True
            self.label = Counter(labels).most_common(1)[0][0]
            return

        self.attribute = best_attr
        self.threshold = best_thresh

        # Split data and recurse
        remaining_attrs = [a for a in attributes if a != best_attr]
        if best_thresh is None:  # Categorical
            unique_vals = data[best_attr].unique()
            for val, subset_labels in zip(unique_vals, best_subsets):
                subset_data = data[data[best_attr] == val]
                self.children[val] = Node(subset_data, remaining_attrs, depth + 1, max_depth, min_samples)
        else:  # Numeric
            # Tranform decimal threshold to integer if possible for better readability 
            if isinstance(best_thresh, float):
                best_thresh = math.ceil(best_thresh)

            left_data = data[data[best_attr] <= best_thresh]
            right_data = data[data[best_attr] > best_thresh]
            self.children['<=' + str(best_thresh)] = Node(left_data, remaining_attrs, depth + 1, max_depth, min_samples)
            self.children['>' + str(best_thresh)] = Node(right_data, remaining_attrs, depth + 1, max_depth, min_samples)

    def print_tree(self, indent=0):
        # If violated then fill with red color, if satisfied then fill with green color
        prefix = "  " * indent
        if self.is_leaf:
            #if self.label == "Satisfied":
                #print(f"{prefix}\033[92mLeaf: {self.label} (samples={self.samples}, entropy={round(self.entropy, 4)})\033[0m")
            #elif self.label == "Violated":
                #print(f"{prefix}\033[91mLeaf: {self.label} (samples={self.samples}, entropy={round(self.entropy, 4)})\033[0m")
            print(f"{prefix}Leaf: {self.label} (samples={self.samples}, entropy={round(self.entropy, 4)})")
        else:
            print(f"{prefix}Node: {self.attribute} (samples={self.samples}, entropy={round(self.entropy, 4)})")
            for key, child in self.children.items():
                print(f"{prefix}  {key}:")
                child.print_tree(indent + 2)

    def build_graphviz(self, dot):
        """Build Graphviz Digraph recursively"""        

        if self.is_leaf: 
            label = f"""<
            <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2">
            <TR><TD><B>{self.label}</B></TD></TR>
            <TR><TD><B>Entropy: {round(self.entropy, 4)}</B></TD></TR>
            <TR><TD><B>Samples: {self.samples}</B></TD></TR>
            </TABLE>
            >"""

            if self.label == "Satisfied":
                # larger font size and bold for better visibility
                dot.node(str(self.node_id), label=label, style="filled", fillcolor="lightgreen", fontsize="100")
            elif self.label == "Violated":
                dot.node(str(self.node_id), label=label, style="filled", fillcolor="lightcoral", fontsize="100")
            else:
                dot.node(str(self.node_id), label=label, fontsize="100")

        else:
            label = f"""<
            <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2">
            <TR><TD><B>{self.attribute}</B></TD></TR>
            <TR><TD><B>Entropy: {round(self.entropy, 4)}</B></TD></TR>
            <TR><TD><B>Samples: {self.samples}</B></TD></TR>
            </TABLE>
            >"""
            
            dot.node(str(self.node_id), label=label, fontsize="100")

        
        for edge_label, child in self.children.items():   
            
            # Transform edge label to be more descriptive and visually appealing
            safe_label = html.escape(str(edge_label))  # VERY IMPORTANT
            # If the edge label is None or empty, we can assign a default value to avoid issues in Graphviz rendering
            """if edge_label is None or str(edge_label).strip() == "":
                safe_label = "unknown"
            else:
                safe_label = html.escape(str(edge_label))"""

            edge_label_transform = f'<<B>{safe_label}</B>>'  # note spaces
            dot.edge(str(self.node_id), str(child.node_id), label=edge_label_transform, fontsize="100")
            child.build_graphviz(dot)
#------------------------------
def build_decision_trees_function(constraint_id,prepared_df):
    """
    This function builds decision trees based on the event logs and constraints.
    """
    print("Building decision trees based on event logs and constraints...")
    # Placeholder for decision tree building logic
    attributes = prepared_df.columns[:-1] # All columns except the last one
    labels = prepared_df.iloc[:, -1].tolist()
    print("Total cases:", len(labels))
    print("Class distribution:", Counter(labels))
    print("Root Entropy H(S):", round(entropy(labels), 4))
    # -----------------------------
    # Build Decision Tree
    # -----------------------------
    
    print("\nBuilding Decision Tree, please wait...")
    root = Node(prepared_df, attributes, max_depth=5, min_samples=10)  # Adjust parameters as needed

    print("\nDecision Tree Structure:")
    root.print_tree()

    # -----------------------------
    # Visualize Decision Tree with Graphviz
    # -----------------------------

    print("\nGenerating Graphviz visualization...")
    dot = Digraph(
        comment="Decision Tree",
        format="svg"  # change to "png" if you prefer
    )
    dot.attr(rankdir="TB", size="100,100") # Set larger size for better readability
    #dot.attr(fontsize="50")  # Adjust font size for better readability

    root.build_graphviz(dot)

    # Export locally
    #dot.render("try-balanced-cases-random", cleanup=True)
    #print("Visualization saved as 'try-balanced-cases-random.svg'")
    # Also export as PDF
    dot.render(f"{constraint_id}-decision-tree", format="pdf", cleanup=True)
    print(f"Visualization saved as '{constraint_id}-decision-tree.pdf'")
    # Also export as PNG
    #dot.render("try-balanced-cases-random", format="png", cleanup=True)
    #print("Visualization saved as 'try-balanced-cases-random.png'")
    # Extract the  each unique path from the root node to the leaf node where label is "Satisfied" and then return it as the activation condition
    def collect_satisfied_paths(node, path, results):
        if node.is_leaf:
            if node.label == "Satisfied":
                results.append(list(path))
            return

        for edge_label, child in node.children.items():
            next_path = path + [(node.attribute, edge_label)]
            collect_satisfied_paths(child, next_path, results)

    satisfied_paths = []
    collect_satisfied_paths(root, [], satisfied_paths)

    unique_paths = []
    seen_paths = set()
    for path in satisfied_paths:
        normalized_path = tuple(path)
        if normalized_path not in seen_paths:
            seen_paths.add(normalized_path)
            unique_paths.append(normalized_path)

    def format_path_as_condition(path):
        conditions = []
        for attribute, edge_label in path:
            if attribute is None:
                continue

            if str(edge_label).startswith("<="):
                conditions.append(f"{attribute} <= {edge_label[2:]}")
            elif str(edge_label).startswith(">"):
                conditions.append(f"{attribute} > {edge_label[1:]}")
            else:
                value_repr = repr(edge_label) if isinstance(edge_label, str) else str(edge_label)
                conditions.append(f"{attribute} == {value_repr}")

        return " AND ".join(conditions) if conditions else "root"

    activation_conditions = [format_path_as_condition(path) for path in unique_paths]

    # Return the activation conditions for further use in the conformance analysis
    return activation_conditions
    
