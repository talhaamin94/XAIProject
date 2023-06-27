import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
#import graphviz
import torch_geometric.transforms as T
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.explain import Explainer,GNNExplainer
from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph
#from torch_geometric.explain.metric import fidelity, groundtruth_metrics
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# dataset = CoraFull(root='.', name="Pubmed")







dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=300, num_edges=5),
    motif_generator='house',
    num_motifs=80,
    transform=T.Constant(),
)
data = dataset[0]
print("number of node features:",data.num_node_features)
print("number of nodes:", data.num_nodes)
print("number of features:", data.num_features)

idx = torch.arange(data.num_nodes)
#creating train_test splits
train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=data.y)
print("Training set:",len(train_idx))
print("Test set:",len(test_idx))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = GCN(data.num_node_features, hidden_channels=20, num_layers=3,
            out_channels=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    train_correct = int((pred[train_idx] == data.y[train_idx]).sum())
    train_acc = train_correct / train_idx.size(0)

    test_correct = int((pred[test_idx] == data.y[test_idx]).sum())
    test_acc = test_correct / test_idx.size(0)

    return train_acc, test_acc


pbar = tqdm(range(1, 2001))
for epoch in pbar:
    loss = train()
    if epoch == 1 or epoch % 200 == 0:
        train_acc, test_acc = test()
        pbar.set_description(f'Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                             f'Test: {test_acc:.4f}')
pbar.close()
model.eval()


subgraph_count = 0
for explanation_type in ['phenomenon','model']:
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


    
    # Explanation ROC AUC over all test nodes:
    targets, preds = [], []
    node_indices = range(400, data.num_nodes, 5)
    for node_index in tqdm(node_indices, leave=False, desc='Train Explainer'):
        target = data.y if explanation_type == 'phenomenon' else None
        explanation = explainer(data.x, data.edge_index, index=node_index,
                                target=target)

        _,_, _, hard_edge_mask = k_hop_subgraph(node_index, num_hops=3,
                                              edge_index=data.edge_index)
        targets.append(data.edge_mask[hard_edge_mask].cpu())
        preds.append(explanation.edge_mask[hard_edge_mask].cpu())

        
        
        # VISUALIZATION CODE HERE
        # Create a NetworkX graph from the subgraph
        subgraph_nodes = torch.unique(data.edge_index[:, hard_edge_mask])
        subgraph = data.subgraph(subgraph_nodes)

        
        subgraph_count += 1
        if(subgraph_count<=5):

            G = nx.Graph()
            x = subgraph.num_nodes
            G.add_nodes_from(range(x))
            edge_index = subgraph.edge_index.cpu().numpy()
            G.add_edges_from(edge_index.T)
            target_class = data.y[node_index].item()

            # Visualize the NetworkX graph
            plt.figure(figsize=(8, 8))
            pos = nx.spring_layout(G, seed=42)
            plt.title(f'Subgraph around node {node_index} (Explanation Type: {explanation_type})\nPredicted Class: {target_class}')
            # Determine edge colors based on the number of connections
            edge_colors = ['red' if G.degree[node] <= 3 else 'black' for node in G.nodes]
            node_colors = ['green' if G.degree[node] > 3 else 'yellow' for node in G.nodes]
            # Draw the graph with different edge colors
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, edge_color=edge_colors)
            nx.draw_networkx_edges(G, pos, width=2.0, edge_color='red')
            
            title = f'Subgraph around node {node_index} Explanation Type: {explanation_type} Predicted Class: {target_class}'

            # Remove special characters from the title
            title = ''.join(c for c in title if c.isalnum() or c.isspace())
            # Save the graph with the generated title name
            plt.savefig(f'{title}.png')

            #plt.show()
            

    auc = roc_auc_score(torch.cat(targets), torch.cat(preds))
    print(f'Mean ROC AUC (explanation type {explanation_type:10}): {auc:.4f}')

