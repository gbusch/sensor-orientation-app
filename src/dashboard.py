"""
Interactive dashboard for visualizing sensor data transformations and quality metrics.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json

from data_simulator import DataSimulator, SensorData
from transformation import VehicleFrameTransformer, TransformationResult
from quality_metrics import QualityAssessment, RealTimeQualityMonitor


class DashboardData:
    """Container for dashboard data"""
    
    def __init__(self):
        self.sensor_data: List[SensorData] = []
        self.transformation_results: List[TransformationResult] = []
        self.quality_report = None
        self.real_time_monitor = RealTimeQualityMonitor()
        
    def load_data(self, duration: float = 60.0, orientation_type: str = "changing"):
        """Generate and load data for dashboard"""
        print("Generating sensor data...")
        simulator = DataSimulator(duration=duration)
        self.sensor_data = simulator.generate_complete_dataset(orientation_type)
        
        print("Transforming data...")
        transformer = VehicleFrameTransformer()
        self.transformation_results = transformer.transform_sensor_data(self.sensor_data)
        
        print("Assessing quality...")
        assessor = QualityAssessment()
        self.quality_report = assessor.assess_transformation_quality(
            self.sensor_data, self.transformation_results
        )
        
        print("Data loaded successfully!")


# Global data container
dashboard_data = DashboardData()


def create_sensor_data_plots() -> go.Figure:
    """Create plots for raw sensor data"""
    
    if not dashboard_data.sensor_data:
        return go.Figure()
    
    timestamps = [data.timestamp for data in dashboard_data.sensor_data]
    
    # Accelerometer data
    accel_x = [data.accelerometer[0] for data in dashboard_data.sensor_data]
    accel_y = [data.accelerometer[1] for data in dashboard_data.sensor_data]
    accel_z = [data.accelerometer[2] for data in dashboard_data.sensor_data]
    
    # Gyroscope data
    gyro_x = [data.gyroscope[0] for data in dashboard_data.sensor_data]
    gyro_y = [data.gyroscope[1] for data in dashboard_data.sensor_data]
    gyro_z = [data.gyroscope[2] for data in dashboard_data.sensor_data]
    
    # GPS velocity
    gps_vx = [data.gps_velocity[0] for data in dashboard_data.sensor_data]
    gps_vy = [data.gps_velocity[1] for data in dashboard_data.sensor_data]
    gps_speed = [np.linalg.norm(data.gps_velocity[:2]) for data in dashboard_data.sensor_data]
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Accelerometer (Phone Frame)', 'Gyroscope (Phone Frame)',
                       'GPS Velocity', 'GPS Speed',
                       'GPS Accuracy', 'Sensor Overview'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accelerometer
    fig.add_trace(go.Scatter(x=timestamps, y=accel_x, name='Accel X', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=accel_y, name='Accel Y', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=accel_z, name='Accel Z', line=dict(color='blue')), row=1, col=1)
    
    # Gyroscope
    fig.add_trace(go.Scatter(x=timestamps, y=gyro_x, name='Gyro X', line=dict(color='red'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=timestamps, y=gyro_y, name='Gyro Y', line=dict(color='green'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=timestamps, y=gyro_z, name='Gyro Z', line=dict(color='blue'), showlegend=False), row=1, col=2)
    
    # GPS Velocity
    fig.add_trace(go.Scatter(x=timestamps, y=gps_vx, name='GPS Vx', line=dict(color='red'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=gps_vy, name='GPS Vy', line=dict(color='green'), showlegend=False), row=2, col=1)
    
    # GPS Speed
    fig.add_trace(go.Scatter(x=timestamps, y=gps_speed, name='Speed', line=dict(color='purple'), showlegend=False), row=2, col=2)
    
    # GPS Accuracy
    gps_accuracy = [data.gps_accuracy for data in dashboard_data.sensor_data]
    fig.add_trace(go.Scatter(x=timestamps, y=gps_accuracy, name='GPS Accuracy', line=dict(color='orange'), showlegend=False), row=3, col=1)
    
    # Sensor magnitude overview
    accel_mag = [np.linalg.norm(data.accelerometer) for data in dashboard_data.sensor_data]
    gyro_mag = [np.linalg.norm(data.gyroscope) for data in dashboard_data.sensor_data]
    fig.add_trace(go.Scatter(x=timestamps, y=accel_mag, name='|Accel|', line=dict(color='red'), showlegend=False), row=3, col=2)
    fig.add_trace(go.Scatter(x=timestamps, y=gyro_mag, name='|Gyro|', line=dict(color='blue'), showlegend=False), row=3, col=2)
    
    fig.update_layout(height=800, title_text="Raw Sensor Data")
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Angular Velocity (rad/s)", row=1, col=2)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", row=2, col=2)
    fig.update_yaxes(title_text="Accuracy (m)", row=3, col=1)
    fig.update_yaxes(title_text="Magnitude", row=3, col=2)
    
    return fig


def create_transformation_plots() -> go.Figure:
    """Create plots for transformation results"""
    
    if not dashboard_data.transformation_results:
        return go.Figure()
    
    timestamps = [result.timestamp for result in dashboard_data.transformation_results]
    
    # Vehicle frame acceleration
    vehicle_accel_x = [result.vehicle_acceleration[0] for result in dashboard_data.transformation_results]
    vehicle_accel_y = [result.vehicle_acceleration[1] for result in dashboard_data.transformation_results]
    vehicle_accel_z = [result.vehicle_acceleration[2] for result in dashboard_data.transformation_results]
    
    # Estimated orientation
    roll = [result.estimated_orientation[0] * 180/np.pi for result in dashboard_data.transformation_results]
    pitch = [result.estimated_orientation[1] * 180/np.pi for result in dashboard_data.transformation_results]
    yaw = [result.estimated_orientation[2] * 180/np.pi for result in dashboard_data.transformation_results]
    
    # Quality scores
    quality_scores = [result.quality_score for result in dashboard_data.transformation_results]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Vehicle Frame Acceleration', 'Estimated Phone Orientation',
                       'Quality Scores', 'Acceleration Components'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Vehicle acceleration
    fig.add_trace(go.Scatter(x=timestamps, y=vehicle_accel_x, name='Forward', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=vehicle_accel_y, name='Lateral', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=vehicle_accel_z, name='Vertical', line=dict(color='blue')), row=1, col=1)
    
    # Orientation
    fig.add_trace(go.Scatter(x=timestamps, y=roll, name='Roll', line=dict(color='red'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=timestamps, y=pitch, name='Pitch', line=dict(color='green'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=timestamps, y=yaw, name='Yaw', line=dict(color='blue'), showlegend=False), row=1, col=2)
    
    # Quality scores
    fig.add_trace(go.Scatter(x=timestamps, y=quality_scores, name='Quality', line=dict(color='purple'), showlegend=False), row=2, col=1)
    
    # Acceleration magnitude and direction
    accel_magnitude = [np.linalg.norm(result.vehicle_acceleration) for result in dashboard_data.transformation_results]
    forward_component = [abs(result.vehicle_acceleration[0]) for result in dashboard_data.transformation_results]
    lateral_component = [abs(result.vehicle_acceleration[1]) for result in dashboard_data.transformation_results]
    
    fig.add_trace(go.Scatter(x=timestamps, y=accel_magnitude, name='Total', line=dict(color='black'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=timestamps, y=forward_component, name='|Forward|', line=dict(color='red'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=timestamps, y=lateral_component, name='|Lateral|', line=dict(color='green'), showlegend=False), row=2, col=2)
    
    fig.update_layout(height=600, title_text="Transformation Results")
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Angle (degrees)", row=1, col=2)
    fig.update_yaxes(title_text="Quality Score", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=2)
    
    return fig


def create_quality_metrics_plot() -> go.Figure:
    """Create quality metrics visualization"""
    
    if not dashboard_data.transformation_results or not dashboard_data.quality_report:
        return go.Figure()
    
    # Quality components
    components = ['Temporal Consistency', 'Physical Plausibility', 'Sensor Reliability', 'Transformation Stability']
    scores = [
        dashboard_data.quality_report.temporal_consistency,
        dashboard_data.quality_report.physical_plausibility,
        dashboard_data.quality_report.sensor_reliability,
        dashboard_data.quality_report.transformation_stability
    ]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],  # Close the polygon
        theta=components + [components[0]],
        fill='toself',
        name='Quality Metrics',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f"Quality Assessment (Overall: {dashboard_data.quality_report.overall_score:.3f})"
    )
    
    return fig


def create_gps_trajectory_plot() -> go.Figure:
    """Create GPS trajectory visualization"""
    
    if not dashboard_data.sensor_data:
        return go.Figure()
    
    # Extract GPS positions
    lats = [data.gps_position[0] for data in dashboard_data.sensor_data]
    lons = [data.gps_position[1] for data in dashboard_data.sensor_data]
    timestamps = [data.timestamp for data in dashboard_data.sensor_data]
    speeds = [np.linalg.norm(data.gps_velocity[:2]) for data in dashboard_data.sensor_data]
    
    fig = go.Figure()
    
    # Color by speed
    fig.add_trace(go.Scatter(
        x=lons, y=lats,
        mode='markers+lines',
        marker=dict(
            size=8,
            color=speeds,
            colorscale='Viridis',
            colorbar=dict(title="Speed (m/s)"),
            showscale=True
        ),
        line=dict(width=2),
        text=[f"Time: {t:.1f}s<br>Speed: {s:.1f} m/s" for t, s in zip(timestamps, speeds)],
        hovertemplate='%{text}<extra></extra>',
        name='GPS Trajectory'
    ))
    
    fig.update_layout(
        title="Vehicle GPS Trajectory",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        showlegend=False
    )
    
    return fig


def create_detailed_metrics_table() -> html.Div:
    """Create detailed metrics table"""
    
    if not dashboard_data.quality_report:
        return html.Div("No quality report available")
    
    metrics = dashboard_data.quality_report.detailed_metrics
    
    table_data = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_value = f"{value:.3f}"
        else:
            formatted_value = str(value)
        
        table_data.append({
            'Metric': key.replace('_', ' ').title(),
            'Value': formatted_value
        })
    
    df = pd.DataFrame(table_data)
    
    return html.Div([
        html.H4("Detailed Quality Metrics"),
        html.Table([
            html.Thead([
                html.Tr([html.Th(col) for col in df.columns])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(df.iloc[i][col]) for col in df.columns
                ]) for i in range(len(df))
            ])
        ], className="table table-striped")
    ])


def create_recommendations_panel() -> html.Div:
    """Create recommendations panel"""
    
    if not dashboard_data.quality_report:
        return html.Div("No recommendations available")
    
    recommendations = dashboard_data.quality_report.recommendations
    
    return html.Div([
        html.H4("Quality Recommendations"),
        html.Ul([
            html.Li(rec) for rec in recommendations
        ])
    ])


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'])

app.layout = html.Div([
    html.Div([
        html.H1("Vehicle Reference Frame Transformation Dashboard", className="text-center mb-4"),
        
        # Control panel
        html.Div([
            html.Div([
                html.Label("Simulation Duration (seconds):"),
                dcc.Slider(
                    id='duration-slider',
                    min=10, max=120, step=10, value=60,
                    marks={i: str(i) for i in range(10, 121, 20)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], className="col-md-6"),
            
            html.Div([
                html.Label("Phone Orientation Type:"),
                dcc.Dropdown(
                    id='orientation-dropdown',
                    options=[
                        {'label': 'Fixed', 'value': 'fixed'},
                        {'label': 'Slowly Changing', 'value': 'slowly_changing'},
                        {'label': 'Dynamic Changes', 'value': 'changing'}
                    ],
                    value='changing'
                )
            ], className="col-md-6")
        ], className="row mb-4"),
        
        html.Div([
            html.Button("Generate New Data", id="generate-button", className="btn btn-primary me-2"),
            html.Button("Export Results", id="export-button", className="btn btn-secondary"),
        ], className="mb-4"),
        
        # Status indicator
        html.Div(id="status-indicator", className="alert alert-info"),
        
        # Main content tabs
        dcc.Tabs(id="main-tabs", value='sensor-data', children=[
            dcc.Tab(label='Raw Sensor Data', value='sensor-data'),
            dcc.Tab(label='Transformation Results', value='transformation'),
            dcc.Tab(label='Quality Assessment', value='quality'),
            dcc.Tab(label='GPS Trajectory', value='trajectory'),
        ]),
        
        html.Div(id='tab-content', className="mt-4")
        
    ], className="container-fluid")
])


@app.callback(
    [Output('status-indicator', 'children'),
     Output('tab-content', 'children')],
    [Input('generate-button', 'n_clicks'),
     Input('main-tabs', 'value')],
    [dash.dependencies.State('duration-slider', 'value'),
     dash.dependencies.State('orientation-dropdown', 'value')]
)
def update_dashboard(n_clicks, active_tab, duration, orientation_type):
    ctx = dash.callback_context
    
    # Generate new data if button clicked
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'generate-button.n_clicks':
        try:
            dashboard_data.load_data(duration, orientation_type)
            status = f"✅ Data generated successfully! {len(dashboard_data.sensor_data)} samples, Quality Score: {dashboard_data.quality_report.overall_score:.3f}"
        except Exception as e:
            status = f"❌ Error generating data: {str(e)}"
            return status, html.Div("Error loading data")
    else:
        if not dashboard_data.sensor_data:
            dashboard_data.load_data(duration, orientation_type)
        status = f"📊 Current dataset: {len(dashboard_data.sensor_data)} samples, Quality Score: {dashboard_data.quality_report.overall_score:.3f}"
    
    # Update tab content
    if active_tab == 'sensor-data':
        content = dcc.Graph(figure=create_sensor_data_plots())
    elif active_tab == 'transformation':
        content = dcc.Graph(figure=create_transformation_plots())
    elif active_tab == 'quality':
        content = html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=create_quality_metrics_plot())
                ], className="col-md-6"),
                html.Div([
                    create_detailed_metrics_table()
                ], className="col-md-6")
            ], className="row"),
            html.Div([
                create_recommendations_panel()
            ], className="row mt-4")
        ])
    elif active_tab == 'trajectory':
        content = dcc.Graph(figure=create_gps_trajectory_plot())
    else:
        content = html.Div("Select a tab to view content")
    
    return status, content


@app.callback(
    Output('export-button', 'n_clicks'),
    Input('export-button', 'n_clicks')
)
def export_results(n_clicks):
    if n_clicks and dashboard_data.sensor_data:
        # Export data to JSON
        simulator = DataSimulator()
        simulator.save_dataset(dashboard_data.sensor_data, "../data/exported_sensor_data.json")
        
        # Export transformation results
        results_data = []
        for result in dashboard_data.transformation_results:
            results_data.append({
                'timestamp': result.timestamp,
                'vehicle_acceleration': result.vehicle_acceleration.tolist(),
                'estimated_orientation': result.estimated_orientation.tolist(),
                'quality_score': result.quality_score,
                'confidence_metrics': result.confidence_metrics
            })
        
        with open('../data/exported_transformation_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print("Results exported to ../data/")
    
    return None


def run_dashboard(host='0.0.0.0', port=52739, debug=False):
    """Run the dashboard application"""
    print(f"Starting dashboard at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard(debug=True)