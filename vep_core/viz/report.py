"""
VEP Visualization Module
========================
Generates interactive "Glass Brain" dashboards with clinical telemetry.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class VEPReport:
    """Clinical reporting engine."""
    
    @staticmethod
    def generate_dashboard(cortex, mapping, time, data, x0_values, onset_times, 
                          labels, full_labels=None, output_path="vep_report.html"):
        """
        Create the full HTML dashboard.
        """
        if full_labels is None:
            full_labels = labels
            
        print(f"[Report] Generating Glass Brain Dashboard at {output_path}...")
        
        vertices, triangles, _ = cortex
        n_regions = data.shape[1]
        
        # --- 1. Setup Figure with Subplots ---
        # Row 1: 3D Brain (70% height)
        # Row 2: 2D Time Series (30% height)
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            specs=[[{'type': 'scene'}], [{'type': 'xy'}]],
            vertical_spacing=0.05,
            subplot_titles=("3D Seizure Propagation (Glass Brain)", "Regional Activity (mV)")
        )
        
        # --- 2. Visual Elements (3D) ---
        
        # A. The Ghost Anatomy (Point Cloud) - Trace 0
        rng = np.random.RandomState(42)
        idx_glass = rng.choice(len(vertices), 4000, replace=False)
        vx, vy, vz = vertices[idx_glass].T
        
        trace_glass = go.Scatter3d(
            x=vx, y=vy, z=vz,
            mode='markers',
            marker=dict(size=2, color='grey', opacity=0.1, symbol='circle'),
            name='Anatomy (Ghost)',
            hoverinfo='none'
        )
        fig.add_trace(trace_glass, row=1, col=1)

        # B. The Active Nodes (Dynamic Scatter) - Trace 1
        # Pre-compute layout for nodes
        # Calculate centroids
        region_centers = np.zeros((n_regions, 3))
        for r in range(n_regions):
            # mapping is array where value is region idx
            # We want vertices belonging to region r
            # TVB mapping: vertex_idx -> region_idx
            # region_mapping is (N_verts,)
            # Wait, loader.load_cortex returns region_mapping
            
            # Simple centroid calculation
            mask = (mapping == r)
            if np.sum(mask) > 0:
                region_centers[r] = np.mean(vertices[mask], axis=0)
            else:
                region_centers[r] = [0, 0, 0] # Fallback
                
        rx, ry, rz = region_centers.T
        
        # Pre-compute Tooltips (Rich HTML)
        node_texts = []
        for i in range(n_regions):
            name = full_labels[i]
            code = labels[i]
            x0 = x0_values[i]
            onset = onset_times[i]
            
            # Status
            if x0 > -2.0:
                status = "<span style='color:#ff4444; font-weight:bold'>SZ ZONE (EZ)</span>"
            elif onset > 0:
                status = "<span style='color:#ffaa00'>Propagated (PZ)</span>"
            else:
                status = "<span style='color:#88cc88'>Healthy</span>"
                
            text = (
                f"<b style='font-size:16px'>{name}</b><br>"
                f"<span style='color:#888; font-size:12px'>ID: {code}</span>"
                f"<hr style='margin:5px 0; border:0; border-top:1px solid #444'>"
                f"<div style='min-width: 180px'>"
                f"  <div style='display:flex; justify-content:space-between'><span>Status:</span> {status}</div>"
                f"  <div style='display:flex; justify-content:space-between'><span>Excitability (x0):</span> <b>{x0:.3f}</b></div>"
                f"  <div style='display:flex; justify-content:space-between'><span>Seizure Onset:</span> <b>{onset:.0f} ms</b></div>"
                f"</div>"
            )
            node_texts.append(text)
            
        def get_node_trace(frame_idx):
            current_vals = data[frame_idx] # (N,)
            
            # Dynamic Sizing & Coloring
            # Sigmoid normalization for size
            # x1 ranges [-2, 2] usually.
            norm_act = (current_vals + 2.0) / 4.0 # 0 to 1
            norm_act = np.clip(norm_act, 0.1, 1.0)
            sizes = 5 + 20 * (norm_act**2)
            
            return go.Scatter3d(
                x=rx, y=ry, z=rz,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=current_vals,
                    colorscale='Magma',
                    cmin=-2.0, cmax=1.0,
                    opacity=0.9
                ),
                text=node_texts,
                hoverinfo='text',
                name='Active Regions'
            )
            
        # Initial State
        dataset_node_trace = get_node_trace(0)
        fig.add_trace(dataset_node_trace, row=1, col=1)
        
        # --- 3. Visual Elements (2D Time Series) ---
        # Plot Top 5 Relevant Regions
        # Priority: EZ regions, then Early Onset regions
        priority_indices = []
        
        # 1. Add EZ
        ez_idx = np.where(x0_values > -2.0)[0]
        priority_indices.extend(ez_idx)
        
        # 2. Add Early Onset (Propagated)
        prop_idx = np.where((onset_times > 0) & (x0_values <= -2.0))[0]
        # Sort by onset time
        sorted_prop = prop_idx[np.argsort(onset_times[prop_idx])]
        priority_indices.extend(sorted_prop[:5]) # Top 5 propagated
        
        # Unique and limit
        unique_indices = []
        seen = set()
        for idx in priority_indices:
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)
                
        # Limit to 8 traces to avoid clutter
        unique_indices = unique_indices[:8]
        
        # Add traces
        colors = ['red', 'orange', 'yellow', 'cyan', 'magenta', 'lime', 'white', 'blue']
        for i, idx in enumerate(unique_indices):
            color = colors[i % len(colors)]
            lbl = labels[idx]
            fig.add_trace(
                go.Scatter(
                    x=time, y=data[:, idx],
                    mode='lines',
                    name=lbl,
                    line=dict(width=1.5, color=color),
                    opacity=0.8
                ),
                row=2, col=1
            )
            
        # --- 4. Animation Frames (Downsampled) ---
        downsample_anim = 5
        frames = []
        # Only animate the 3D trace (Trace 1 in Figure -> actually index might be different with subplots)
        # In subplots, traces are flattened list.
        # Trace 0: Ghost (3D)
        # Trace 1: Nodes (3D)
        # Traces 2+: Lines (2D)
        # We only want to animate Trace 1.
        
        skip = 20 # 100ms per frame if dt=0.05 -> 20 steps = 1ms. 
        # Simulation duration 4000ms. 4000/skip frames.
        # Let's say we want 100 frames total? 
        step_stride = max(1, len(time) // 200) 
        
        for k in range(0, len(time), step_stride):
            frame_node_trace = get_node_trace(k)
            frames.append(go.Frame(
                data=[frame_node_trace], # Only updating trace 1?
                # Plotly update limitation: need to target specific trace indices.
                # If we pass list of length 1, it might update trace 0?
                # We need explicit traces=[1]
                traces=[1], 
                name=str(time[k])
            ))
            
        fig.frames = frames
        
        # --- 5. Layout & Controls ---
        # Update Scene (3D)
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=0, z=0.5)),
                dragmode='orbit',
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                domain=dict(y=[0.35, 1.0]) # Improve 3D space usage
            ),
            xaxis2=dict(title="Time (ms)", color="white", showgrid=True, gridcolor="#333"),
            yaxis2=dict(title="x1 (mV)", color="white", showgrid=True, gridcolor="#333", range=[-2.5, 2.0]),
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=900,
            showlegend=True,
            legend=dict(orientation="h", y=0.32, x=0.5, xanchor="center")
        )
        
        # Sliders/Buttons
        # Need to disable sliders for 2D? No, sliders control frame.
        # Frame matches time.
        fig.update_layout(
            updatemenus=[dict(
                type='buttons', showactive=False, y=0.05, x=0.05,
                buttons=[
                    dict(label='▶ PLAY', method='animate', args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                    dict(label='Ⅱ PAUSE', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
                ]
            )]
        )
        
        # --- 6. Interaction Annotation ---
        fig.add_annotation(
            text="Controls: Use 'Replay' to restart. Click Nodes for Info.",
            xref="paper", yref="paper",
            x=0.05, y=0.02, showarrow=False, xanchor='left',
            font=dict(color="gray", size=12)
        )
        
        # --- 7. Export with Custom JS ---
        CONFIG = {'scrollZoom': True, 'displayModeBar': True, 'responsive': True}
        
        # Force a specific DIV ID
        html_content = fig.to_html(config=CONFIG, include_plotlyjs='cdn', full_html=True, div_id="vep-graph")
        
        # Inject Custom UI and JS
        custom_ui = """
        <style>
            #info-panel {
                position: absolute;
                top: 20px;
                right: 20px;
                width: 320px;
                background: rgba(15, 15, 15, 0.95);
                border: 1px solid #555;
                border-radius: 8px;
                color: #ddd;
                font-family: 'Helvetica Neue', Arial, sans-serif;
                padding: 15px;
                z-index: 9999;
                box-shadow: 0 4px 20px rgba(0,0,0,0.8);
                display: none;
                backdrop-filter: blur(5px);
            }
            #info-panel h3 { margin-top: 0; color: #44aaff; border-bottom: 1px solid #444; padding-bottom: 10px; font-size: 16px; }
            #close-btn { position: absolute; top: 10px; right: 10px; cursor: pointer; color: #888; font-size: 18px; }
            #close-btn:hover { color: white; }
            #panel-content { white-space: normal !important; overflow-wrap: break-word; }
            .plotly-tooltip { display: none !important; } 
        </style>
        
        <div id="info-panel">
            <div id="close-btn" onclick="document.getElementById('info-panel').style.display='none'">✕</div>
            <div id="panel-content">
                <!-- Content injected here -->
            </div>
        </div>

        <script>
            console.log("VEP-Report: Initializing Custom Scripts...");
            window.onload = function() {
                var plotDiv = document.getElementById('vep-graph');
                if (!plotDiv) {
                    console.error("VEP-Report: Graph DIV not found!");
                    return;
                }
                console.log("VEP-Report: Graph DIV found. Attaching click listener.");
                plotDiv.on('plotly_click', function(data){
                    console.log("VEP-Report: Click Detected!", data);
                    if (!data || !data.points || data.points.length === 0) return;
                    
                    var pt = data.points[0];
                    // Only react to 3D Node Trace (Trace 1)
                    // With subplots, trace numbering is global.
                    // Trace 0: Ghost. Trace 1: Nodes. Trace 2+: Lines.
                    if (pt.curveNumber === 1 && pt.text) { 
                        var content = pt.text;
                        var panel = document.getElementById('info-panel');
                        var container = document.getElementById('panel-content');
                        container.innerHTML = "<h3>Region Details</h3>" + content;
                        container.innerHTML = container.innerHTML.replace(/min-width: 180px/g, 'width: 100%');
                        panel.style.display = 'block';
                    }
                });
            };
        </script>
        """
        
        final_html = html_content.replace('</body>', f'{custom_ui}</body>')
        
        with open(output_path, 'w') as f:
            f.write(final_html)
            
        print("[Report] Dashboard saved with Interactive Info Panel (Fixed ID) + 2D Plots.")
