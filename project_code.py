#XAI Project
#Talha Amin, amint, 6944916

import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


def train_model(model, data, train_idx, optimizer):
    """
    Train the GCN model.

    Args:
        model (GCN): The GCN model to be trained.
        data (Data): The input data.
        train_idx (torch.Tensor): Indices of nodes in the training set.
        optimizer (torch.optim.Optimizer): The optimizer for training.

    Returns:
        float: Training loss.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate_model(model, data, train_idx, test_idx):
    """
    Evaluate the GCN model on training and test data.

    Args:
        model (GCN): The trained GCN model.
        data (Data): The input data.
        train_idx (torch.Tensor): Indices of nodes in the training set.
        test_idx (torch.Tensor): Indices of nodes in the test set.

    Returns:
        tuple: A tuple containing train accuracy and test accuracy.
    """
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    train_correct = int((pred[train_idx] == data.y[train_idx]).sum())
    train_acc = train_correct / train_idx.size(0)

    test_correct = int((pred[test_idx] == data.y[test_idx]).sum())
    test_acc = test_correct / test_idx.size(0)

    return train_acc, test_acc


def visualize_subgraph(subgraph, node_index, explanation_type, subgraph_nodes):
    G = to_networkx(subgraph)

    
    """
    Visualize a subgraph around a given node.

    Args:
        subgraph (Data): The subgraph data.
        node_index (int): The index of the central node.
        explanation_type (str): The type of explanation ('phenomenon' or 'model').
    """
    G = nx.Graph()
    x = subgraph.num_nodes
    G.add_nodes_from(range(x))
    edge_index = subgraph.edge_index.cpu().numpy()
    G.add_edges_from(edge_index.T)

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    plt.title(f'Subgraph around node {node_index} (Explanation Type: {explanation_type})')

    # Create a dictionary of node indices to their labels
    labels = {node: str(subgraph_nodes[node].item()) for node in G.nodes}
    node_colors = ['yellow' for node in G.nodes]

    distances = {node: abs(node - node_index) for node in G.nodes}
    closest_nodes = sorted(distances, key=distances.get)[:5]

    node_colors = ['lightgreen' if node in closest_nodes and labels[node] != str(node_index) else 'red' if labels[node] == str(node_index) else 'yellow' for node in G.nodes]

    nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors, edge_color='black', linewidths=1, font_color='black', node_size=1000)

    title = f'Subgraph around node {node_index} Explanation Type: {explanation_type}'
    title = ''.join(c for c in title if c.isalnum() or c.isspace())
    plt.savefig(title+'.png')


def explain_subgraphs(model, data, explanation_type):
    """
    Explain subgraphs using GNNExplainer.

    Args:
        model (GCN): The trained GCN model.
        data (Data): The input data.
        explanation_type (str): The type of explanation ('phenomenon' or 'model').
    """
    subgraph_count = 0
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=300),
        explanation_type=explanation_type,
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    targets, preds = [], []
    node_indices = range(400, data.num_nodes, 5)
    for node_index in tqdm(node_indices, leave=False, desc='Train Explainer'):
        target = data.y if explanation_type == 'phenomenon' else None
        explanation = explainer(data.x, data.edge_index, index=node_index, target=target)
        
        _, _, _, hard_edge_mask = k_hop_subgraph(node_index, num_hops=3, edge_index=data.edge_index)
        targets.append(data.edge_mask[hard_edge_mask].cpu())
        preds.append(explanation.edge_mask[hard_edge_mask].cpu())
        if subgraph_count <= 5:
            subgraph_nodes = torch.unique(data.edge_index[:, hard_edge_mask])
            subgraph = data.subgraph(subgraph_nodes)
            visualize_subgraph(subgraph, node_index, explanation_type, subgraph_nodes)


            subgraph_count += 1

    auc = roc_auc_score(torch.cat(targets), torch.cat(preds))
    print(f'Mean ROC AUC (explanation type {explanation_type:10}): {auc:.4f}')


def main():
    dataset = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=300, num_edges=5),
        motif_generator='house',
        num_motifs=80,
        transform=T.Constant(),
    )
    data = dataset[0]
    print("Number of node features:", data.num_node_features)
    print("Number of nodes:", data.num_nodes)
    print("Number of features:", data.num_features)

    idx = torch.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=data.y)
    print("Training set:", len(train_idx))
    print("Test set:", len(test_idx))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = GCN(data.num_node_features, hidden_channels=20, num_layers=3, out_channels=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)

    pbar = tqdm(range(1, 2001))
    for epoch in pbar:
        loss = train_model(model, data, train_idx, optimizer)
        if epoch == 1 or epoch % 200 == 0:
            train_acc, test_acc = evaluate_model(model, data, train_idx, test_idx)
            pbar.set_description(f'Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    pbar.close()
    model.eval()

    for explanation_type in ['phenomenon', 'model']:
        explain_subgraphs(model, data, explanation_type)


if __name__ == '__main__':
    main()
