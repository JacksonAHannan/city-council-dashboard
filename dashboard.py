import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import json
import numpy as np
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data using relative paths
df = pd.read_excel(os.path.join(BASE_DIR, 'City Council District 2 2022 Data.xlsx'))
gdf = gpd.read_file(os.path.join(BASE_DIR, 'tl_2020_01_vtd20.shp'))

# Build GEOID20 to match shapefile format: state (01) + county (089) + precinct (padded to 6 digits)
df['GEOID20_full'] = '01089' + df['GEOID20'].astype(str).str.zfill(6)

# Convert shapefile GEOID20 to string for matching
gdf['GEOID20'] = gdf['GEOID20'].astype(str)

# Merge data
merged = gdf.merge(df, left_on='GEOID20', right_on='GEOID20_full', how='inner')

# Calculate derived metrics
candidates = ['Little', 'Petters', 'Yell']
merged['Total_Votes'] = merged[candidates].sum(axis=1)

for cand in candidates:
    merged[f'{cand}_pct'] = (merged[cand] / merged['Total_Votes'] * 100).round(1)

# Pairwise comparisons
merged['Little_vs_Petters'] = merged['Little'] - merged['Petters']
merged['Little_vs_Yell'] = merged['Little'] - merged['Yell']
merged['Petters_vs_Yell'] = merged['Petters'] - merged['Yell']

# Percentage margins
merged['Little_vs_Petters_pct'] = ((merged['Little'] - merged['Petters']) / merged['Total_Votes'] * 100).round(1)
merged['Little_vs_Yell_pct'] = ((merged['Little'] - merged['Yell']) / merged['Total_Votes'] * 100).round(1)
merged['Petters_vs_Yell_pct'] = ((merged['Petters'] - merged['Yell']) / merged['Total_Votes'] * 100).round(1)

# Three-way comparison - determine winner in each precinct
def get_winner(row):
    votes = {'Little': row['Little'], 'Petters': row['Petters'], 'Yell': row['Yell']}
    winner = max(votes, key=votes.get)
    return winner

def get_winner_margin(row):
    votes = sorted([row['Little'], row['Petters'], row['Yell']], reverse=True)
    return votes[0] - votes[1]  # margin over second place

merged['Winner'] = merged.apply(get_winner, axis=1)
merged['Winner_Margin'] = merged.apply(get_winner_margin, axis=1)
merged['Winner_Margin_Pct'] = (merged['Winner_Margin'] / merged['Total_Votes'] * 100).round(1)

# Assign numeric codes for coloring (categorical)
winner_map = {'Little': 0, 'Petters': 1, 'Yell': 2}
merged['Winner_Code'] = merged['Winner'].map(winner_map)

# Convert to WGS84 for mapping
merged = merged.to_crs(epsg=4326)

# Calculate centroids for bubble overlay (use projected CRS for accurate centroids)
merged_projected = merged.to_crs(epsg=3857)  # Web Mercator for centroid calculation
centroids = merged_projected.geometry.centroid.to_crs(epsg=4326)
merged['centroid_lat'] = centroids.y
merged['centroid_lon'] = centroids.x

# Normalize bubble sizes
max_votes = merged['Total_Votes'].max()
min_votes = merged['Total_Votes'].min()
merged['bubble_size'] = 10 + 40 * (merged['Total_Votes'] - min_votes) / (max_votes - min_votes)

# Get GeoJSON for Plotly
geojson = json.loads(merged.to_json())

# Map center calculation
center_lat = merged['centroid_lat'].mean()
center_lon = merged['centroid_lon'].mean()

# Candidate colors - Navy, Gold, Magenta for better contrast
candidate_colors = {'Little': '#000080', 'Petters': '#FFD700', 'Yell': '#FF00FF'}

# Helper function to blend white to a color based on intensity (0-1)
def blend_to_white(hex_color, intensity):
    """Blend from white (intensity=0) to hex_color (intensity=1)"""
    # Parse hex color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    # Blend with white (255, 255, 255)
    r_blend = int(255 + (r - 255) * intensity)
    g_blend = int(255 + (g - 255) * intensity)
    b_blend = int(255 + (b - 255) * intensity)
    return f'#{r_blend:02x}{g_blend:02x}{b_blend:02x}'

# Compute winner-colored margin values
# Normalize margins for color intensity (0 to 1)
max_margin = merged['Winner_Margin'].max()
max_margin_pct = merged['Winner_Margin_Pct'].max()

def get_winner_color_by_margin(row, use_pct=False):
    winner = row['Winner']
    base_color = candidate_colors[winner]
    if use_pct:
        intensity = min(row['Winner_Margin_Pct'] / max_margin_pct, 1.0) if max_margin_pct > 0 else 0.5
    else:
        intensity = min(row['Winner_Margin'] / max_margin, 1.0) if max_margin > 0 else 0.5
    # Ensure minimum intensity so colors are visible
    intensity = 0.3 + 0.7 * intensity
    return blend_to_white(base_color, intensity)

merged['Winner_Color_Margin'] = merged.apply(lambda r: get_winner_color_by_margin(r, False), axis=1)
merged['Winner_Color_Margin_Pct'] = merged.apply(lambda r: get_winner_color_by_margin(r, True), axis=1)

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("City Council District 2 - 2022 Election Results", className="text-center mb-4 mt-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("View Options"),
                dbc.CardBody([
                    html.Label("Select View:", className="fw-bold"),
                    dcc.RadioItems(
                        id='view-type',
                        options=[
                            {'label': ' Individual Candidate', 'value': 'individual'},
                            {'label': ' Head-to-Head (2-way)', 'value': 'comparison'},
                            {'label': ' Three-Way Comparison', 'value': 'threeway'}
                        ],
                        value='individual',
                        className="mb-3"
                    ),
                    # Candidate selector (shown for individual view)
                    html.Div(id='candidate-container', children=[
                        html.Label("Select Candidate:", className="fw-bold mt-2"),
                        dcc.Dropdown(
                            id='candidate-select',
                            options=[
                                {'label': 'Little', 'value': 'Little'},
                                {'label': 'Petters', 'value': 'Petters'},
                                {'label': 'Yell', 'value': 'Yell'}
                            ],
                            value='Little',
                            clearable=False
                        )
                    ]),
                    # Comparison selector (shown for head-to-head view)
                    html.Div(id='comparison-container', style={'display': 'none'}, children=[
                        html.Label("Select Comparison:", className="fw-bold mt-2"),
                        dcc.Dropdown(
                            id='comparison-select',
                            options=[
                                {'label': 'Little vs Petters', 'value': 'Little_vs_Petters'},
                                {'label': 'Little vs Yell', 'value': 'Little_vs_Yell'},
                                {'label': 'Petters vs Yell', 'value': 'Petters_vs_Yell'}
                            ],
                            value='Little_vs_Petters',
                            clearable=False
                        )
                    ]),
                    # Metric type selector
                    html.Div(id='metric-container', children=[
                        html.Label("Display Metric:", className="fw-bold mt-3"),
                        dcc.RadioItems(
                            id='metric-type',
                            options=[
                                {'label': ' Vote Count', 'value': 'count'},
                                {'label': ' Vote Percentage', 'value': 'percent'}
                            ],
                            value='percent'
                        )
                    ]),
                    html.Hr(),
                    html.Label("Voter Volume Overlay:", className="fw-bold"),
                    dcc.Checklist(
                        id='show-bubbles',
                        options=[{'label': ' Show vote volume pie charts', 'value': 'show'}],
                        value=[],
                        className="mb-2"
                    ),
                    html.Small("Pie chart size indicates total votes; slices show candidate vote share", 
                              className="text-muted")
                ])
            ]),
            dbc.Card([
                dbc.CardHeader("Legend"),
                dbc.CardBody(id='legend-container')
            ], className="mt-3")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='map', style={'height': '600px'})
                ])
            ])
        ], width=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Precinct Results Table"),
                dbc.CardBody([
                    html.Div(id='results-table')
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Vote Totals Summary"),
                dbc.CardBody([
                    dcc.Graph(id='bar-chart', style={'height': '350px'})
                ])
            ])
        ], width=4)
    ], className="mt-3"),
], fluid=True)


@callback(
    [Output('candidate-container', 'style'),
     Output('comparison-container', 'style'),
     Output('metric-type', 'options'),
     Output('metric-type', 'value')],
    [Input('view-type', 'value')],
    prevent_initial_call=False
)
def update_selector(view_type):
    print(f"update_selector called with view_type={view_type}")
    if view_type == 'individual':
        metric_options = [
            {'label': ' Vote Count', 'value': 'count'},
            {'label': ' Vote Percentage', 'value': 'percent'}
        ]
        return ({'display': 'block'}, {'display': 'none'}, metric_options, 'percent')
    elif view_type == 'comparison':
        metric_options = [
            {'label': ' Vote Margin', 'value': 'count'},
            {'label': ' Margin Percentage', 'value': 'percent'}
        ]
        return ({'display': 'none'}, {'display': 'block'}, metric_options, 'percent')
    else:  # threeway
        metric_options = [
            {'label': ' Winner by precinct', 'value': 'winner'},
            {'label': ' Winner margin (votes)', 'value': 'margin_count'},
            {'label': ' Winner margin (%)', 'value': 'margin_pct'}
        ]
        return ({'display': 'none'}, {'display': 'none'}, metric_options, 'winner')


@callback(
    Output('legend-container', 'children'),
    Input('view-type', 'value')
)
def update_legend(view_type):
    if view_type == 'threeway':
        return html.Div([
            html.Div([
                html.Span("■ ", style={'color': candidate_colors['Little'], 'fontSize': '20px'}),
                html.Span("Little")
            ]),
            html.Div([
                html.Span("■ ", style={'color': candidate_colors['Petters'], 'fontSize': '20px'}),
                html.Span("Petters")
            ]),
            html.Div([
                html.Span("■ ", style={'color': candidate_colors['Yell'], 'fontSize': '20px'}),
                html.Span("Yell")
            ]),
            html.Hr(),
            html.Small("Colors show which candidate won each precinct", className="text-muted")
        ])
    elif view_type == 'comparison':
        return html.Div([
            html.Div([
                html.Span("■ ", style={'color': candidate_colors['Little'], 'fontSize': '20px'}),
                html.Span("Little (Navy) - positive margin")
            ]),
            html.Div([
                html.Span("■ ", style={'color': candidate_colors['Petters'], 'fontSize': '20px'}),
                html.Span("Petters (Gold)")
            ]),
            html.Div([
                html.Span("■ ", style={'color': candidate_colors['Yell'], 'fontSize': '20px'}),
                html.Span("Yell (Magenta)")
            ]),
            html.Hr(),
            html.Small("First candidate's color = positive, Second = negative", className="text-muted")
        ])
    else:
        return html.Div([
            html.Div([
                html.Span("Darker blue = higher vote share/count", className="text-muted")
            ])
        ])


@callback(
    Output('map', 'figure'),
    [Input('view-type', 'value'),
     Input('candidate-select', 'value'),
     Input('comparison-select', 'value'),
     Input('metric-type', 'value'),
     Input('show-bubbles', 'value')]
)
def update_map(view_type, candidate, comparison, metric_type, show_bubbles):
    print(f"update_map called: view_type={view_type}, metric_type={metric_type}")
    show_volume = 'show' in show_bubbles if show_bubbles else False
    
    if view_type == 'individual':
        if candidate is None:
            candidate = 'Little'
        
        # Handle invalid metric_type when switching from threeway
        if metric_type not in ['percent', 'count']:
            metric_type = 'percent'
        
        if metric_type == 'percent':
            color_col = f'{candidate}_pct'
            title = f'{candidate} - Vote Share (%)'
            color_label = 'Vote %'
        else:
            color_col = candidate
            title = f'{candidate} - Vote Count'
            color_label = 'Votes'
        
        # Create color scale from white to candidate's color
        cand_color = candidate_colors[candidate]
        individual_scale = [[0, '#FFFFFF'], [1, cand_color]]
        
        fig = px.choropleth_map(
            merged,
            geojson=geojson,
            locations=merged.index,
            color=color_col,
            hover_name='Precinct Name',
            hover_data={
                'Little': True,
                'Petters': True,
                'Yell': True,
                'Total_Votes': True,
                'Little_pct': ':.1f',
                'Petters_pct': ':.1f',
                'Yell_pct': ':.1f'
            },
            color_continuous_scale=individual_scale,
            map_style='carto-positron',
            zoom=10,
            center={'lat': center_lat, 'lon': center_lon},
            opacity=0.7,
            labels={color_col: color_label}
        )
        
    elif view_type == 'comparison':
        if comparison is None:
            comparison = 'Little_vs_Petters'
        
        # Handle invalid metric_type when switching from threeway
        if metric_type not in ['percent', 'count']:
            metric_type = 'percent'
        
        if metric_type == 'percent':
            color_col = f'{comparison}_pct'
            title = f'{comparison.replace("_", " ")} - Margin (%)'
            color_label = 'Margin %'
        else:
            color_col = comparison
            title = f'{comparison.replace("_", " ")} - Vote Margin'
            color_label = 'Vote Margin'
        
        cand1, cand2 = comparison.split('_vs_')
        
        # Create custom color scale using candidate colors
        color1 = candidate_colors[cand1]
        color2 = candidate_colors[cand2]
        custom_scale = [[0, color2], [0.5, '#FFFFFF'], [1, color1]]
        
        fig = px.choropleth_map(
            merged,
            geojson=geojson,
            locations=merged.index,
            color=color_col,
            hover_name='Precinct Name',
            hover_data={
                cand1: True,
                cand2: True,
                'Total_Votes': True,
                color_col: ':.1f'
            },
            color_continuous_scale=custom_scale,
            color_continuous_midpoint=0,
            map_style='carto-positron',
            zoom=10,
            center={'lat': center_lat, 'lon': center_lon},
            opacity=0.7,
            labels={color_col: color_label}
        )
        
    else:  # threeway
        if metric_type == 'winner':
            # Categorical coloring by winner
            fig = px.choropleth_map(
                merged,
                geojson=geojson,
                locations=merged.index,
                color='Winner',
                hover_name='Precinct Name',
                hover_data={
                    'Little': True,
                    'Petters': True,
                    'Yell': True,
                    'Total_Votes': True,
                    'Winner_Margin': True,
                    'Winner_Margin_Pct': ':.1f'
                },
                color_discrete_map=candidate_colors,
                map_style='carto-positron',
                zoom=10,
                center={'lat': center_lat, 'lon': center_lon},
                opacity=0.7
            )
            title = 'Three-Way: Winner by Precinct'
        elif metric_type == 'margin_count':
            # Use Winner_Color_Margin for intensity-based coloring
            fig = px.choropleth_map(
                merged,
                geojson=geojson,
                locations=merged.index,
                color='Winner_Color_Margin',
                hover_name='Precinct Name',
                hover_data={
                    'Winner': True,
                    'Little': True,
                    'Petters': True,
                    'Yell': True,
                    'Total_Votes': True,
                    'Winner_Margin': True,
                    'Winner_Margin_Pct': ':.1f'
                },
                color_discrete_map={c: c for c in merged['Winner_Color_Margin'].unique()},
                map_style='carto-positron',
                zoom=10,
                center={'lat': center_lat, 'lon': center_lon},
                opacity=0.85
            )
            # Hide the legend since colors are hex codes
            fig.update_layout(showlegend=False)
            
            # Add simple text labels
            fig.add_trace(go.Scattermap(
                lat=list(merged['centroid_lat']),
                lon=list(merged['centroid_lon']),
                mode='text',
                text=[f"+{m}" for m in merged['Winner_Margin']],
                textfont=dict(size=11, color='black'),
                textposition='middle center',
                hoverinfo='skip',
                showlegend=False
            ))
            title = 'Three-Way: Winner Margin (votes) - Intensity = Margin Size'
        elif metric_type == 'margin_pct':
            # Use Winner_Color_Margin_Pct for intensity-based coloring
            fig = px.choropleth_map(
                merged,
                geojson=geojson,
                locations=merged.index,
                color='Winner_Color_Margin_Pct',
                hover_name='Precinct Name',
                hover_data={
                    'Winner': True,
                    'Little': True,
                    'Petters': True,
                    'Yell': True,
                    'Total_Votes': True,
                    'Winner_Margin': True,
                    'Winner_Margin_Pct': ':.1f'
                },
                color_discrete_map={c: c for c in merged['Winner_Color_Margin_Pct'].unique()},
                map_style='carto-positron',
                zoom=10,
                center={'lat': center_lat, 'lon': center_lon},
                opacity=0.85
            )
            # Hide the legend since colors are hex codes
            fig.update_layout(showlegend=False)
            
            # Add simple text labels
            fig.add_trace(go.Scattermap(
                lat=list(merged['centroid_lat']),
                lon=list(merged['centroid_lon']),
                mode='text',
                text=[f"+{m:.1f}%" for m in merged['Winner_Margin_Pct']],
                textfont=dict(size=11, color='black'),
                textposition='middle center',
                hoverinfo='skip',
                showlegend=False
            ))
            title = 'Three-Way: Winner Margin (%) - Intensity = Margin Size'
        else:
            # Fallback to winner view for any other metric_type
            fig = px.choropleth_map(
                merged,
                geojson=geojson,
                locations=merged.index,
                color='Winner',
                hover_name='Precinct Name',
                hover_data={
                    'Little': True,
                    'Petters': True,
                    'Yell': True,
                    'Total_Votes': True,
                    'Winner_Margin': True,
                    'Winner_Margin_Pct': ':.1f'
                },
                color_discrete_map=candidate_colors,
                map_style='carto-positron',
                zoom=10,
                center={'lat': center_lat, 'lon': center_lon},
                opacity=0.7
            )
            title = 'Three-Way: Winner by Precinct'
    
    # Add pie chart overlay for vote volume if enabled
    if show_volume:
        import math
        # Add pie charts as polygon traces for each precinct
        for idx, row in merged.iterrows():
            total = row['Total_Votes']
            if total == 0:
                continue
                
            little_pct = row['Little'] / total
            petters_pct = row['Petters'] / total
            yell_pct = row['Yell'] / total
            
            # Scale size based on total votes - smaller radius for better fit
            r = 0.002 + 0.004 * (row['bubble_size'] - 10) / 40  # radius in degrees
            
            lat_center = row['centroid_lat']
            lon_center = row['centroid_lon']
            
            hover_text = (f"<b>{row['Precinct Name']}</b><br>"
                         f"Total: {total}<br>"
                         f"Little: {row['Little']} ({row['Little_pct']}%)<br>"
                         f"Petters: {row['Petters']} ({row['Petters_pct']}%)<br>"
                         f"Yell: {row['Yell']} ({row['Yell_pct']}%)")
            
            n_points = 20  # points per slice for smooth edges
            pcts = [little_pct, petters_pct, yell_pct]
            pie_colors = [candidate_colors['Little'], candidate_colors['Petters'], candidate_colors['Yell']]
            labels = ['Little', 'Petters', 'Yell']
            
            cumulative = 0
            for i, (pct, color, label) in enumerate(zip(pcts, pie_colors, labels)):
                if pct > 0.001:  # Skip very small slices
                    start_angle = cumulative * 360 - 90  # Start from top
                    end_angle = start_angle + pct * 360
                    
                    theta = np.linspace(math.radians(start_angle), math.radians(end_angle), n_points)
                    
                    # Correct for longitude distortion at latitude
                    lon_scale = 1 / math.cos(math.radians(lat_center))
                    
                    # Create polygon points for pie slice
                    lats = [lat_center] + [lat_center + r * math.sin(t) for t in theta] + [lat_center]
                    lons = [lon_center] + [lon_center + r * lon_scale * math.cos(t) for t in theta] + [lon_center]
                    
                    # Use Scattermap for px.choropleth_map figures (uses map not mapbox)
                    fig.add_trace(go.Scattermap(
                        lat=lats,
                        lon=lons,
                        mode='lines',
                        fill='toself',
                        fillcolor=color,
                        line=dict(color='white', width=0.5),
                        opacity=0.9,
                        text=hover_text,
                        hoverinfo='text',
                        name=label,
                        showlegend=False
                    ))
                cumulative += pct
    
    fig.update_layout(
        title=title,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        showlegend=show_volume
    )
    
    return fig


@callback(
    Output('results-table', 'children'),
    [Input('view-type', 'value'),
     Input('candidate-select', 'value'),
     Input('comparison-select', 'value')]
)
def update_table(view_type, candidate, comparison):
    if view_type == 'threeway':
        table_df = merged[['Precinct Name', 'Little', 'Petters', 'Yell', 'Total_Votes', 
                           'Winner', 'Winner_Margin', 'Winner_Margin_Pct']].copy()
        table_df.columns = ['Precinct', 'Little', 'Petters', 'Yell', 'Total', 
                            'Winner', 'Margin', 'Margin %']
        table_df = table_df.sort_values('Total', ascending=False)
    else:
        table_df = merged[['Precinct Name', 'Little', 'Petters', 'Yell', 'Total_Votes', 
                           'Little_pct', 'Petters_pct', 'Yell_pct']].copy()
        table_df.columns = ['Precinct', 'Little', 'Petters', 'Yell', 'Total', 
                            'Little %', 'Petters %', 'Yell %']
        
        if view_type == 'individual' and candidate:
            table_df = table_df.sort_values(f'{candidate} %', ascending=False)
        elif view_type == 'comparison' and comparison:
            cand1, cand2 = comparison.split('_vs_')
            table_df = table_df.sort_values(cand1, ascending=False)
        else:
            table_df = table_df.sort_values('Total', ascending=False)
    
    return dbc.Table.from_dataframe(
        table_df.round(1),
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm"
    )


@callback(
    Output('bar-chart', 'figure'),
    Input('view-type', 'value')
)
def update_bar(view_type):
    totals = {
        'Little': merged['Little'].sum(),
        'Petters': merged['Petters'].sum(),
        'Yell': merged['Yell'].sum()
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(totals.keys()),
            y=list(totals.values()),
            marker_color=[candidate_colors['Little'], candidate_colors['Petters'], candidate_colors['Yell']],
            text=list(totals.values()),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Total Votes by Candidate',
        xaxis_title='Candidate',
        yaxis_title='Total Votes',
        margin={"r": 20, "t": 40, "l": 50, "b": 40}
    )
    
    return fig


# Expose server for Gunicorn
server = app.server

if __name__ == '__main__':
    print(f"\nSuccessfully matched {len(merged)} precincts")
    print(f"Total votes - Little: {merged['Little'].sum()}, Petters: {merged['Petters'].sum()}, Yell: {merged['Yell'].sum()}")
    print(f"\nPrecincts won by each candidate:")
    print(merged['Winner'].value_counts().to_string())
    print("\nStarting dashboard server...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True)
