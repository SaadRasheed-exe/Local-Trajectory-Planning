import math
import json
from typing import Dict, List, Any
from bokeh.plotting import figure, show, output_file
from bokeh.models import (
    ColumnDataSource, HoverTool, CustomJS, 
    Slider, LinearColorMapper, ColorBar, LabelSet, Arrow, OpenHead, TextInput, Label
)
from bokeh.layouts import column, row
from bokeh.palettes import Reds256

from models.models import PlanResult, PredictedEnvironment, GoalRegion
from omegaconf import DictConfig

def _get_bbox_corners(x: float, y: float, yaw: float, length: float, width: float) -> tuple[list[float], list[float]]:
    """
    Helper function: Calculates the corner points of a rotated rectangle for Bokeh (Patches).
    Uses a 2D rotation matrix to shift the local corners into the global coordinate frame.
    """
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    hw, hl = width / 2.0, length / 2.0
    corners_local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    xs, ys = [], []
    for lx, ly in corners_local:
        xs.append(x + (lx * cos_y - ly * sin_y))
        ys.append(y + (lx * sin_y + ly * cos_y))
    return xs, ys

def visualize_search_tree(
    plan_result: PlanResult, 
    pred_env: PredictedEnvironment, 
    goal_region: GoalRegion, 
    cfg: DictConfig,
    vehicle_cfg: DictConfig,
    output_filename: str = "planner_debug.html"
):
    """
    Generates an interactive HTML dashboard to analyze the Hybrid A* search tree.
    All data is pre-computed in Python and passed to JavaScript callbacks for 
    high-performance client-side rendering without a Python backend server.
    """
    if plan_result.debug_root_node is None:
        print("No root node found. Visualization aborted.")
        return

    # -------------------------------------------------------------------------
    # 1. Data extraction of nodes and velocity vectors
    # -------------------------------------------------------------------------
    # We flatten the tree structure into dictionary arrays for the Bokeh ColumnDataSource
    nodes_data = {
        'id': [], 'x': [], 'y': [], 'timestamp': [], 'cost': [], 'cost_str': [], 
        'fill_alpha': [], 'line_alpha': [], 'line_width': [], 'size': [], 
        'details_str': [], 'parent_x': [], 'parent_y': [], 'parent_id': [], 'parent_dist': [],
        'ego_box_xs': [], 'ego_box_ys': [], 'vec_x0': [], 'vec_y0': [], 'vec_x1': [], 'vec_y1': []
    }
    edges_data = {'xs': [], 'ys': []}
    vec_map = {}
    
    stack = [plan_result.debug_root_node]
    timestamps_set = set()
    max_cost = 0.0

    # Base timestamp of the simulation step
    t0 = plan_result.debug_root_node.state_stamped.timestamp
    # Distance from the rear axle to the geometric center for accurate bounding box drawing
    ego_center_offset = vehicle_cfg.length / 2.0 - vehicle_cfg.rear_to_wheel
    
    while stack:
        node = stack.pop()
        state = node.state_stamped.state
        ts = node.state_stamped.timestamp

        # Separate infinite costs: 
        # NaN is required so the ColorMapper renders it as white instead of crashing.
        # "inf" is a string representation for the UI tooltips.
        raw_cost = node.total_cost
        if math.isinf(raw_cost) or math.isnan(raw_cost):
            safe_cost = float('nan') 
            display_cost = "inf"
        else:
            safe_cost = float(raw_cost)
            display_cost = f"{safe_cost:.2f}"
            if safe_cost > max_cost:
                max_cost = safe_cost

        timestamps_set.add(ts)
        
        # Format the detailed hover string containing kinematics and motion primitives
        details = f"Node ID: {node.id}\n"
        details += f"Timestamp: {ts} ms\n"
        details += f"Total Cost: {display_cost}\n"
        details += "-"*25 + "\n"
        details += f"Pos: x={state.pos.x:.2f} m, y={state.pos.y:.2f} m\n"
        
        v_ms = math.hypot(state.velocity.x, state.velocity.y)
        details += f"v: {v_ms:.2f} m/s ({v_ms * 3.6:.2f} km/h)\n"
        
        a_ms = math.hypot(state.acceleration.x, state.acceleration.y)
        details += f"a: {a_ms:.2f} m/s²\n"
        
        details += f"yaw: {state.yaw:.2f} rad\n"
        details += f"steer: {state.steering_angle:.2f} rad\n"
        
        details += "-"*25 + "\n"
        if hasattr(node, 'motion_primitive') and node.motion_primitive:
            details += f"MP Steer: {node.motion_primitive.steering_angle:.2f} rad\n"
            details += f"MP Accel: {node.motion_primitive.acceleration:.2f} m/s²\n"
        else:
            details += "MP: None (Root Node)\n"
            
        details += "-"*25 + "\n"
        if node.detailed_costs:
            for k, v in node.detailed_costs.items():
                if math.isinf(v) or math.isnan(v):
                    details += f"{k}: inf\n"
                else:
                    details += f"{k}: {v:.2f}\n"
        else:
            details += "Costs: N/A"

        # Calculate the geometric center for this specific node to draw the bounding box correctly
        cx = state.pos.x + ego_center_offset * math.cos(state.yaw)
        cy = state.pos.y + ego_center_offset * math.sin(state.yaw)
        b_xs, b_ys = _get_bbox_corners(cx, cy, state.yaw, vehicle_cfg.length, vehicle_cfg.width)

        # Append basic node data
        nodes_data['id'].append(node.id)
        nodes_data['x'].append(state.pos.x)
        nodes_data['y'].append(state.pos.y)
        nodes_data['timestamp'].append(ts)
        nodes_data['cost'].append(safe_cost)        
        nodes_data['cost_str'].append(display_cost) 
        nodes_data['details_str'].append(details)
        
        # Store pre-calculated geometric data for JavaScript highlighting (drawn instantly on click)
        nodes_data['ego_box_xs'].append(b_xs)
        nodes_data['ego_box_ys'].append(b_ys)
        
        vec_scale = 1.0 
        nodes_data['vec_x0'].append(state.pos.x)
        nodes_data['vec_y0'].append(state.pos.y)
        nodes_data['vec_x1'].append(state.pos.x + state.velocity.x * vec_scale)
        nodes_data['vec_y1'].append(state.pos.y + state.velocity.y * vec_scale)
        
        # Store parent information to draw edges (connections) between nodes
        if node.parent is not None:
            px = node.parent.state_stamped.state.pos.x
            py = node.parent.state_stamped.state.pos.y
            dist = math.hypot(state.pos.x - px, state.pos.y - py)
            
            nodes_data['parent_x'].append(px)
            nodes_data['parent_y'].append(py)
            nodes_data['parent_id'].append(node.parent.id)
            nodes_data['parent_dist'].append(dist)
            
            edges_data['xs'].append([px, state.pos.x])
            edges_data['ys'].append([py, state.pos.y])
        else:
            nodes_data['parent_x'].append(float('nan'))
            nodes_data['parent_y'].append(float('nan'))
            nodes_data['parent_id'].append(-1)
            nodes_data['parent_dist'].append(float('nan'))
        
        # Initial highlighting: Only the nodes at t0 are fully visible and clickable initially
        if ts == t0:
            nodes_data['fill_alpha'].append(1.0)
            nodes_data['line_alpha'].append(1.0)
            nodes_data['line_width'].append(2.0)
            nodes_data['size'].append(14)
        else:
            nodes_data['fill_alpha'].append(0.2)
            nodes_data['line_alpha'].append(0.2)
            nodes_data['line_width'].append(0.5)
            nodes_data['size'].append(6)

        # Store velocity vectors for the global time-step view (grouped by timestamp)
        if ts not in vec_map:
            vec_map[ts] = {'x0': [], 'y0': [], 'x1': [], 'y1': []}
        vec_map[ts]['x0'].append(state.pos.x)
        vec_map[ts]['y0'].append(state.pos.y)
        vec_map[ts]['x1'].append(state.pos.x + state.velocity.x * vec_scale)
        vec_map[ts]['y1'].append(state.pos.y + state.velocity.y * vec_scale)

        stack.extend(node.children)

    sorted_times = sorted(list(timestamps_set))
    if not sorted_times:
        sorted_times = [0]

    # -------------------------------------------------------------------------
    # 2. Data preparation for dynamic objects and lanes
    # -------------------------------------------------------------------------
    obj_t0_data = {'xs': [], 'ys': [], 'cx': [], 'cy': [], 'text': []}
    obj_pred_map = {} 

    # Process object predictions and map them to their corresponding timestamps
    for obj_list in pred_env.objects.values():
        for obj_stamped in obj_list:
            ts = obj_stamped.timestamp
            s = obj_stamped.state
            xs, ys = _get_bbox_corners(s.pos.x, s.pos.y, s.yaw, s.length, s.width)
            obj_label = f"Object {s.id}"
            
            if ts == t0:
                obj_t0_data['xs'].append(xs)
                obj_t0_data['ys'].append(ys)
                obj_t0_data['cx'].append(s.pos.x)
                obj_t0_data['cy'].append(s.pos.y)
                obj_t0_data['text'].append(obj_label)
                
            if ts not in obj_pred_map:
                obj_pred_map[ts] = {'xs': [], 'ys': [], 'cx': [], 'cy': [], 'text': []}
            obj_pred_map[ts]['xs'].append(xs)
            obj_pred_map[ts]['ys'].append(ys)
            obj_pred_map[ts]['cx'].append(s.pos.x)
            obj_pred_map[ts]['cy'].append(s.pos.y)
            obj_pred_map[ts]['text'].append(obj_label)

    lane_bounds_xs, lane_bounds_ys = [], []
    lane_center_xs, lane_center_ys = [], []
    
    # Calculate lane boundaries using the normal vector perpendicular to the centerline
    if hasattr(pred_env, 'lanes') and pred_env.lanes:
        for lane in pred_env.lanes:
            cx, cy = [], []
            lx, ly = [], []
            rx, ry = [], []
            half_w = lane.width / 2.0
            
            for pt, yaw in lane.centerline:
                cx.append(pt.x)
                cy.append(pt.y)
                nx = math.cos(yaw + math.pi/2)
                ny = math.sin(yaw + math.pi/2)
                lx.append(pt.x + nx * half_w)
                ly.append(pt.y + ny * half_w)
                rx.append(pt.x - nx * half_w)
                ry.append(pt.y - ny * half_w)
                
            lane_center_xs.append(cx)
            lane_center_ys.append(cy)
            lane_bounds_xs.append(lx)
            lane_bounds_ys.append(ly)
            lane_bounds_xs.append(rx)
            lane_bounds_ys.append(ry)

    # Calculate the corner points of the goal region polygon
    gx, gy = _get_bbox_corners(goal_region.center.x, goal_region.center.y, goal_region.yaw, goal_region.length, goal_region.width)

    # -------------------------------------------------------------------------
    # 3. Bokeh Figure Setup
    # -------------------------------------------------------------------------
    output_file(output_filename, title="Hybrid A* Debugger")
    
    safe_max_cost = max(max_cost, 0.001)
    mapper = LinearColorMapper(palette=Reds256, low=0, high=safe_max_cost, nan_color="white")

    p = figure(
        title="Hybrid A* Tree Visualization", 
        match_aspect=True, sizing_mode="stretch_both",
        tools="pan,wheel_zoom,box_zoom,reset,tap", 
        active_scroll="wheel_zoom"
    )

    # Draw lane boundaries and centerlines
    p.multi_line(lane_bounds_xs, lane_bounds_ys, color="dimgray", line_width=2, legend_label="Lane Boundaries")
    p.multi_line(lane_center_xs, lane_center_ys, color="silver", line_width=1, line_dash="dashed", legend_label="Centerlines")

    # Draw the goal region and its text label
    p.patch(gx, gy, color="limegreen", alpha=0.3, line_color="darkgreen", line_width=2, legend_label="Goal Region")
    goal_label = Label(
        x=goal_region.center.x, y=goal_region.center.y, text="Goal Region",
        text_align="center", text_baseline="middle", 
        text_font_size="10pt", text_font_style="bold", text_color="darkgreen"
    )
    p.add_layout(goal_label)

    # Draw the search tree edges (all connections between nodes)
    source_edges = ColumnDataSource(edges_data)
    p.multi_line('xs', 'ys', source=source_edges, color="gray", alpha=0.3, line_width=1)

    # Setup data source for the selected edge (highlighted parent connection on click)
    edge_hl_source = ColumnDataSource(data=dict(xs=[], ys=[]))
    p.multi_line('xs', 'ys', source=edge_hl_source, color="gold", line_width=4, alpha=0.8)

    # Draw objects (both initial t0 and predicted) including labels
    source_t0_objs = ColumnDataSource(obj_t0_data)
    p.patches('xs', 'ys', source=source_t0_objs, color="blue", alpha=0.1, line_width=2, line_dash="dotted", legend_label="Objects t0")
    t0_labels = LabelSet(
        x='cx', y='cy', text='text', source=source_t0_objs,
        text_align="center", text_baseline="middle", text_font_size="8pt", text_color="blue", text_alpha=0.6
    )
    p.add_layout(t0_labels)

    source_pred_objs = ColumnDataSource(data={'xs': [], 'ys': [], 'cx': [], 'cy': [], 'text': []})
    p.patches('xs', 'ys', source=source_pred_objs, color="blue", alpha=0.5, line_width=2, legend_label="Objects Predicted")
    pred_labels = LabelSet(
        x='cx', y='cy', text='text', source=source_pred_objs,
        text_align="center", text_baseline="middle", text_font_size="9pt", text_font_style="bold", text_color="darkblue"
    )
    p.add_layout(pred_labels)

    # Draw velocity vectors (orange arrows for the general time-step view)
    source_vectors = ColumnDataSource(data=vec_map[t0])
    arrow_glyph = Arrow(
        end=OpenHead(line_color="darkorange", line_width=2, size=6),
        x_start='x0', y_start='y0', x_end='x1', y_end='y1',
        line_color="darkorange", line_width=2,
        source=source_vectors
    )
    p.add_layout(arrow_glyph)

    # Selected ego vehicle represented as a blue bounding box
    ego_box_source = ColumnDataSource(data=dict(xs=[], ys=[]))
    p.patches('xs', 'ys', source=ego_box_source, color="dodgerblue", alpha=0.4, line_color="darkblue", line_width=2, legend_label="Selected Ego Node")

    # Selected velocity vector represented as a blue arrow
    selected_vector_source = ColumnDataSource(data=dict(x0=[], y0=[], x1=[], y1=[]))
    sel_arrow_glyph = Arrow(
        end=OpenHead(line_color="dodgerblue", line_width=3, size=8),
        x_start='x0', y_start='y0', x_end='x1', y_end='y1',
        line_color="dodgerblue", line_width=3,
        source=selected_vector_source
    )
    p.add_layout(sel_arrow_glyph)

    # Separate renderer for highlighting the parent node (golden frame)
    parent_source = ColumnDataSource(data=dict(x=[], y=[]))
    p.scatter('x', 'y', size=20, fill_color="gold", fill_alpha=0.8, line_color="black", line_width=2, source=parent_source)

    # Draw the actual tree nodes as a scatter plot
    source_nodes = ColumnDataSource(nodes_data)
    nodes_renderer = p.scatter(
        x='x', y='y', size={'field': 'size'}, source=source_nodes,
        fill_color={'field': 'cost', 'transform': mapper},
        fill_alpha={'field': 'fill_alpha'}, 
        line_color="black",
        line_alpha={'field': 'line_alpha'},
        line_width={'field': 'line_width'}
    )

    color_bar = ColorBar(color_mapper=mapper, label_standoff=12, border_line_color=None, location=(0,0))
    p.add_layout(color_bar, 'right')

    # -------------------------------------------------------------------------
    # 4. Interactivity: Tooltips, Labels, and Click Events
    # -------------------------------------------------------------------------
    hover = HoverTool(renderers=[nodes_renderer], tooltips=[
        ("ID", "@id"),
        ("Time", "@timestamp ms"),
        ("Cost", "@cost_str")
    ])
    p.add_tools(hover)

    # Info label box for displaying detailed node metadata on click
    info_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    labels_info = LabelSet(
        x='x', y='y', text='text', source=info_source,
        x_offset=15, y_offset=-10,
        background_fill_color="white", background_fill_alpha=0.9,
        border_line_color="black", text_font_size="10pt"
    )
    p.add_layout(labels_info)

    # Labels for selected node ID, parent ID, and edge distance
    edge_label_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))
    labels_edge = LabelSet(
        x='x', y='y', text='text', source=edge_label_source,
        text_align="center", text_baseline="middle", y_offset=12,
        text_font_style="bold", text_color="black",
        background_fill_color="white", background_fill_alpha=0.8
    )
    p.add_layout(labels_edge)
    
    # JavaScript Callback for Node Click (Tap Event)
    # Extracts the pre-computed node geometry and forces a redraw of the highlight layers
    source_nodes.selected.js_on_change('indices', CustomJS(args=dict(
        source=source_nodes, 
        info_source=info_source, 
        parent_source=parent_source,
        edge_hl_source=edge_hl_source,
        edge_label_source=edge_label_source,
        ego_box_source=ego_box_source,
        selected_vector_source=selected_vector_source
    ), code="""
        const inds = cb_obj.indices;
        if (inds.length > 0) {
            const idx = inds[0];
            const nx = source.data['x'][idx];
            const ny = source.data['y'][idx];
            const n_id = source.data['id'][idx];
            
            // 1. Update info label box
            info_source.data['x'] = [nx];
            info_source.data['y'] = [ny];
            info_source.data['text'] = [source.data['details_str'][idx]];
            
            // 2. Update parent highlights and distance text
            const px = source.data['parent_x'][idx];
            const py = source.data['parent_y'][idx];
            
            if (!isNaN(px)) {
                parent_source.data['x'] = [px];
                parent_source.data['y'] = [py];
                
                edge_hl_source.data['xs'] = [[nx, px]];
                edge_hl_source.data['ys'] = [[ny, py]];
                
                const p_id = source.data['parent_id'][idx];
                const dist = source.data['parent_dist'][idx];
                const mid_x = (nx + px) / 2.0;
                const mid_y = (ny + py) / 2.0;
                
                edge_label_source.data['x'] = [nx, px, mid_x];
                edge_label_source.data['y'] = [ny, py, mid_y];
                edge_label_source.data['text'] = [String(n_id), String(p_id), dist.toFixed(2) + "m"];
            } else {
                parent_source.data['x'] = [];
                parent_source.data['y'] = [];
                edge_hl_source.data['xs'] = [];
                edge_hl_source.data['ys'] = [];
                edge_label_source.data['x'] = [];
                edge_label_source.data['y'] = [];
                edge_label_source.data['text'] = [];
            }
            
            // 3. Set ego bounding box & vector highlight using pre-calculated arrays
            ego_box_source.data['xs'] = [source.data['ego_box_xs'][idx]];
            ego_box_source.data['ys'] = [source.data['ego_box_ys'][idx]];
            
            selected_vector_source.data['x0'] = [source.data['vec_x0'][idx]];
            selected_vector_source.data['y0'] = [source.data['vec_y0'][idx]];
            selected_vector_source.data['x1'] = [source.data['vec_x1'][idx]];
            selected_vector_source.data['y1'] = [source.data['vec_y1'][idx]];
            
        } else {
            // Clear all highlights and labels if clicked outside a node
            info_source.data['x'] = [];
            info_source.data['y'] = [];
            info_source.data['text'] = [];
            parent_source.data['x'] = [];
            parent_source.data['y'] = [];
            edge_hl_source.data['xs'] = [];
            edge_hl_source.data['ys'] = [];
            edge_label_source.data['x'] = [];
            edge_label_source.data['y'] = [];
            edge_label_source.data['text'] = [];
            
            ego_box_source.data['xs'] = [];
            ego_box_source.data['ys'] = [];
            
            selected_vector_source.data['x0'] = [];
            selected_vector_source.data['y0'] = [];
            selected_vector_source.data['x1'] = [];
            selected_vector_source.data['y1'] = [];
        }
        
        // Emit changes to trigger a canvas redraw
        info_source.change.emit();
        parent_source.change.emit();
        edge_hl_source.change.emit();
        edge_label_source.change.emit();
        ego_box_source.change.emit();
        selected_vector_source.change.emit();
    """))

    # -------------------------------------------------------------------------
    # 5. UI Controls: Time Slider and Node Search
    # -------------------------------------------------------------------------
    slider = Slider(start=sorted_times[0], end=sorted_times[-1], value=sorted_times[0], step=cfg.planner.dt_sim, title="Timestamp [ms]")
    search_input = TextInput(title="Search Node ID:", placeholder="Enter ID...")

    # JavaScript Callback for the Time Slider
    # Handles dynamic opacity changes to fade out past/future nodes
    slider_callback = CustomJS(
        args=dict(
            source_nodes=source_nodes, 
            source_objs=source_pred_objs,
            source_vectors=source_vectors,
            info_source=info_source,
            parent_source=parent_source,
            edge_hl_source=edge_hl_source,
            edge_label_source=edge_label_source,
            ego_box_source=ego_box_source,
            selected_vector_source=selected_vector_source,
            obj_map_json=json.dumps(obj_pred_map),
            vec_map_json=json.dumps(vec_map)
        ), 
        code="""
        const t_sel = cb_obj.value;
        
        // 1. Update node opacities based on the selected timestamp
        const times = source_nodes.data['timestamp'];
        const fill_alpha = source_nodes.data['fill_alpha'];
        const line_alpha = source_nodes.data['line_alpha'];
        const line_width = source_nodes.data['line_width'];
        const size = source_nodes.data['size'];
        
        for (let i = 0; i < times.length; i++) {
            if (times[i] === t_sel) {
                fill_alpha[i] = 1.0;
                line_alpha[i] = 1.0;
                line_width[i] = 2.0;
                size[i] = 14;
            } else {
                fill_alpha[i] = 0.2;
                line_alpha[i] = 0.2;
                line_width[i] = 0.5;
                size[i] = 6;
            }
        }
        source_nodes.change.emit();
        
        // 2. Hide popups and highlights when time changes to avoid ghost artifacts
        info_source.data['x'] = [];
        info_source.data['y'] = [];
        info_source.data['text'] = [];
        info_source.change.emit();
        
        parent_source.data['x'] = [];
        parent_source.data['y'] = [];
        parent_source.change.emit();
        
        edge_hl_source.data['xs'] = [];
        edge_hl_source.data['ys'] = [];
        edge_hl_source.change.emit();
        
        edge_label_source.data['x'] = [];
        edge_label_source.data['y'] = [];
        edge_label_source.data['text'] = [];
        edge_label_source.change.emit();
        
        ego_box_source.data['xs'] = [];
        ego_box_source.data['ys'] = [];
        ego_box_source.change.emit();
        
        selected_vector_source.data['x0'] = [];
        selected_vector_source.data['y0'] = [];
        selected_vector_source.data['x1'] = [];
        selected_vector_source.data['y1'] = [];
        selected_vector_source.change.emit();
        
        // 3. Update predicted objects (including labels) using JSON mappings
        const str_t = t_sel.toString();
        const obj_map = JSON.parse(obj_map_json);
        if (obj_map[str_t]) {
            source_objs.data['xs'] = obj_map[str_t]['xs'];
            source_objs.data['ys'] = obj_map[str_t]['ys'];
            source_objs.data['cx'] = obj_map[str_t]['cx'];
            source_objs.data['cy'] = obj_map[str_t]['cy'];
            source_objs.data['text'] = obj_map[str_t]['text'];
        } else {
            source_objs.data['xs'] = [];
            source_objs.data['ys'] = [];
            source_objs.data['cx'] = [];
            source_objs.data['cy'] = [];
            source_objs.data['text'] = [];
        }
        source_objs.change.emit();
        
        // 4. Update velocity vectors (the general orange vectors for the active timeframe)
        const vec_map = JSON.parse(vec_map_json);
        if (vec_map[str_t]) {
            source_vectors.data['x0'] = vec_map[str_t]['x0'];
            source_vectors.data['y0'] = vec_map[str_t]['y0'];
            source_vectors.data['x1'] = vec_map[str_t]['x1'];
            source_vectors.data['y1'] = vec_map[str_t]['y1'];
        } else {
            source_vectors.data['x0'] = [];
            source_vectors.data['y0'] = [];
            source_vectors.data['x1'] = [];
            source_vectors.data['y1'] = [];
        }
        source_vectors.change.emit();
    """)
    slider.js_on_change('value', slider_callback)

    # JavaScript Callback for the Search Function
    search_callback = CustomJS(
        args=dict(source_nodes=source_nodes, slider=slider),
        code="""
        const search_val = parseInt(cb_obj.value);
        if (isNaN(search_val)) return;
        
        const ids = source_nodes.data['id'];
        const idx = ids.indexOf(search_val);
        
        if (idx !== -1) {
            // Jump slider to the corresponding time of the found node
            const ts = source_nodes.data['timestamp'][idx];
            slider.value = ts; 
            
            // Select the node (this triggers the tap JS callback automatically to show details)
            source_nodes.selected.indices = [idx];
        } else {
            console.log("ID " + search_val + " not found in tree.");
        }
    """)
    search_input.js_on_change('value', search_callback)

    # -------------------------------------------------------------------------
    # 6. Render Layout
    # -------------------------------------------------------------------------
    controls = row(slider, search_input, sizing_mode="stretch_width")
    layout = column(controls, p, sizing_mode="stretch_both")
    show(layout)