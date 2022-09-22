import torch

from torch_geometric.nn.pool import radius_graph
from typing import Tuple


def graph_changed_nodes(module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
    """ 计算改变了值的nodes，标准是x，即node自身的值（不是edge）
    """
    num_prev_nodes = module.asy_graph.num_nodes
    x_graph = x[:num_prev_nodes]
    different_node_idx = (~torch.isclose(x_graph, module.asy_graph.x)).long()
    different_node_idx = torch.nonzero(torch.sum(different_node_idx, dim=1))[:, 0]
    return x_graph, different_node_idx


def graph_new_nodes(module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
    """ 计算新添加的nodes。因为nodes的储存添加是顺序的，所以直接算中之前nodes num之后的一切就好。
    """
    num_prev_nodes = module.asy_graph.num_nodes
    assert x.size()[0] >= num_prev_nodes, "node deletion is not supported"
    x_graph = x[num_prev_nodes:]
    new_node_idx = torch.arange(num_prev_nodes, x.size()[0], device=x.device, dtype=torch.long).long()
    return x_graph, new_node_idx


def compute_edges(module, pos: torch.Tensor) -> torch.LongTensor:
    """ 以相互距离为基准构建图。以给定半径对每个node做计算。
    """
    return radius_graph(pos, r=module.asy_radius, max_num_neighbors=pos.size()[0])
