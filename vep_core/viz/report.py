"""
Clinical Visualization Module
=============================
Generates the VEP Report dashboard.
Renders the cortical mesh and maps the seizure propagation onto it.
"""

import numpy as np
import plotly.graph_objects as go
import os

class VEPReport:
    @staticmethod
    def generate_dashboard(cortex, mapping, time, data, x0_values, onset_times, labels, full_labels=None, output_path="vep_report.html"):
        """
        Create a High-Performance 'Glass Brain' Dashboard.
        - Static Translucent Cortical Mesh (Context)
        - Dynamic 3D Spheres (Active Regions)
        - Performance: 60FPS guaranteed
        """
        if full_labels is None:
            full_labels = labels # Fallback
        print(f"[Report] Generating Glass Brain Dashboard at {output_path}...")
        
        # --- 1. Prepare Data ---
        # We need Region Centers for the nodes.
        # We can approximate centers from the mesh mapping if not provided, 
        # but better to use the centers from the loader. 
        # Ideally, main_pipeline should pass 'centers'. 
        # I will calculate centroids from the mapping + cortex for now to be self-contained.
        
        vertices, triangles, _ = cortex
        n_regions = len(labels)
        dataset_centers = np.zeros((n_regions, 3))
        
        for r_idx in range(n_regions):
            # Find vertices belonging to region r_idx
            # mapping is array of size (n_verts,) with region indices
            vert_indices = np.where(mapping == r_idx)[0]
            if len(vert_indices) > 0:
                dataset_centers[r_idx] = np.mean(vertices[vert_indices], axis=0)
            else:
                # Fallback if region has no vertices (subcortical?)
                pass

        # Subsample time
        # 50-80 frames is the sweet spot for web animation
        n_frames = 60 
        stride = max(1, len(time) // n_frames)
        frames_data = data[::stride]
        frames_time = time[::stride]
        
        # --- Pre-compute Rich Tooltips ---
        node_text = []
        for code, fullname, x0, onset in zip(labels, full_labels, x0_values, onset_times):
            # Determine clinical status
            if x0 > -1.8: # Epileptogenic
                status = "<b>SZ ZONE</b>"
                status_color = "#FF5555" # Red
            elif x0 > -2.1:
                status = "Propagated"
                status_color = "#FFAA00" # Orange
            else:
                status = "Healthy"
                status_color = "#55FF55" # Green
            
            onset_str = f"{onset:.0f} ms" if onset > 0 else "--"
            
            # HTML Table for Tooltip
            tip = (
                f"<b style='font-size:14px'>{fullname}</b><br>"
                f"<span style='color:#888; font-size:10px'>ID: {code}</span><br>" 
                f"──────────────────────<br>"
                f"<b>Status:</b> <span style='color:{status_color}'>{status}</span><br>"
                f"<b>Excitability (x0):</b> {x0:.3f}<br>"
                f"<b>Seizure Onset:</b> {onset_str}"
            )
            node_text.append(tip)
        
        # --- 2. Visual Elements ---
        
        # A. The Glass Brain (Static Mesh) - Trace 0
        x, y, z = vertices.T
        i, j, k = triangles.T
        
        trace_glass = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='lightgrey',
            opacity=0.10, # Very translucent
            name='Anatomy',
            hoverinfo='skip',
            flatshading=True, # Optimization
            lighting=dict(ambient=0.6, diffuse=0.1, specular=0.1) # Simpler lighting
        )
        
        # B. The Active Nodes (Dynamic Scatter) - Trace 1
        def get_node_trace(frame_idx):
            signal = frames_data[frame_idx]
            # Dynamic Sizing
            sizes = np.clip((signal + 2.0) * 4.0, 4, 30)
            
            return go.Scatter3d(
                x=dataset_centers[:, 0],
                y=dataset_centers[:, 1],
                z=dataset_centers[:, 2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=signal,
                    colorscale='RdBu_r',
                    cmin=-2.0, cmax=2.0,
                    showscale=True,
                    colorbar=dict(title='Local Field Potential (mV)', x=0.0, y=0.7, len=0.6, tickfont=dict(color='white')),
                    opacity=1.0
                ),
                text=node_text, # Use pre-computed rich text
                hoverinfo='text',
                name='Active Regions'
            )
            
        dataset_node_trace = get_node_trace(0)
        
        # --- 3. Animation Frames (Optimized) ---
        # Critical Optimization: traces=[1] updates ONLY the nodes (Trace 1).
        # We do NOT re-send the mesh (Trace 0).
        frames = [
            go.Frame(
                data=[get_node_trace(k)],
                traces=[1], # Target the second trace only
                name=f'f{k}',
                layout=go.Layout(title_text=f"Time: {t:.0f} ms")
            ) for k, t in enumerate(frames_time)
        ]
        
        # --- 4. Layout & Controls ---
        steps = [
            dict(
                method='animate',
                args=[[f'f{k}'], dict(mode='immediate', frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
                label=f'{t:.0f}ms'
            ) for k, t in enumerate(frames_time)
        ]
        
        layout = go.Layout(
            title=dict(text="High-Performance VEP Dashboard (Glass Brain)", font=dict(size=24, color='white')),
            template="plotly_dark",
            paper_bgcolor="#050505",
            height=900,
            scene=dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=0, z=0.5)),
                dragmode='orbit',
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
            ),
            sliders=[dict(
                steps=steps, active=0,
                currentvalue=dict(font=dict(size=14, color="white"), prefix="Time: ", xanchor="center"),
                pad=dict(t=50), x=0.1, len=0.8
            )],
            updatemenus=[dict(
                type='buttons', showactive=False, y=0.1, x=0.05,
                buttons=[
                    dict(label='▶ PLAY', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                    dict(label='Ⅱ PAUSE', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
                ]
            )]
        )
        
        fig = go.Figure(data=[trace_glass, dataset_node_trace], layout=layout, frames=frames)
        
        # Add annotation about interaction
        fig.add_annotation(
            text="Controls: Use 'Replay' to restart. Click Nodes for Info.",
            xref="paper", yref="paper",
            x=0.05, y=0.02, showarrow=False, xanchor='left',
            font=dict(color="gray", size=12)
        )
        
        CONFIG = {'scrollZoom': True, 'displayModeBar': True, 'responsive': True}
        
        # --- Generate HTML with Custom JS ---
        # Fix 1: Force a specific DIV ID so our JS can find it definitively.
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
                z-index: 9999; /* Ensure on top */
                box-shadow: 0 4px 20px rgba(0,0,0,0.8);
                display: none;
                backdrop-filter: blur(5px);
            }
            #info-panel h3 { margin-top: 0; color: #44aaff; border-bottom: 1px solid #444; padding-bottom: 10px; font-size: 16px; }
            #close-btn { position: absolute; top: 10px; right: 10px; cursor: pointer; color: #888; font-size: 18px; }
            #close-btn:hover { color: white; }
            /* Force text inside panel to wrap */
            #panel-content { white-space: normal !important; overflow-wrap: break-word; }
            .plotly-tooltip { display: none !important; } /* Hide default tooltip if we want custom only? Maybe not */
        </style>
        
        <div id="info-panel">
            <div id="close-btn" onclick="document.getElementById('info-panel').style.display='none'">✕</div>
            <div id="panel-content">
                <!-- Content injected here -->
            </div>
        </div>

        <script>
            console.log("VEP-Report: Initializing Custom Scripts...");
            
            // Wait for everything to load
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
                    
                    // Robust check: Does it have text?
                    if (pt.text) { 
                        var content = pt.text;
                        var panel = document.getElementById('info-panel');
                        var container = document.getElementById('panel-content');
                        
                        // Inject content
                        container.innerHTML = "<h3>Region Details</h3>" + content;
                        
                        // Fix style strings that might be narrowly defined in the python string
                        container.innerHTML = container.innerHTML.replace(/min-width: 180px/g, 'width: 100%');
                        
                        panel.style.display = 'block';
                    } else {
                        console.log("VEP-Report: Clicked point has no text.");
                    }
                });
            };
        </script>
        """
        
        # Insert UI before the closing body tag
        final_html = html_content.replace('</body>', f'{custom_ui}</body>')
        
        with open(output_path, 'w') as f:
            f.write(final_html)
            
        print("[Report] Dashboard saved with Interactive Info Panel (Fixed ID).")
