from models.models import *
from anytree import NodeMixin
from planner.motion_primitives import MotionPrimitive


class StateNode(NodeMixin):
    def __init__(self,
                 id: int,
                 state_stamped: EgoStateStamped,
                 node_cost: float,
                 heuristic_cost: float, 
                 path_cost: float,
                 total_cost: float,
                 detailed_costs: Dict[str, float],
                 goal_region_reached: bool,
                 node_depth: float,
                 parent: NodeMixin = None,
                 motion_primitive: MotionPrimitive = None):
        super().__init__()
        self.id = id
        self.state_stamped = state_stamped
        self.node_cost = node_cost
        self.heuristic_cost = heuristic_cost
        self.path_cost = path_cost
        self.total_cost = total_cost
        self.detailed_costs = detailed_costs
        self.goal_region_reached = goal_region_reached
        self.node_depth =  node_depth
        self.parent = parent
        self.motion_primitive = motion_primitive

    def __lt__(self, other):
        if self.total_cost == other.total_cost:
            return self.id < other.id
        return self.total_cost < other.total_cost
    
    def __repr__(self) -> str:
        parent_id = self.parent.id if self.parent is not None else "None"
        
        return (
            f"<StateNode(id={self.id}, depth={self.depth})\n"
            f" ├─ Target Reached : {self.goal_region_reached}\n"
            f" ├─ Parent ID      : {parent_id}\n"
            f" ├─ Costs          : Total={self.total_cost:.3f} | Path={self.path_cost:.3f} | "
            f"Node={self.node_cost:.3f} | Heuristic={self.heuristic_cost:.3f}\n"
            f" ├─ Detailed Costs : {self.detailed_costs}\n"
            f" ├─ State          : {self.state_stamped}\n"
            f" └─ Primitive      : {self.motion_primitive}>"
        )
    