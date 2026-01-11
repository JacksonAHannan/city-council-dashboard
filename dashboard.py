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
df = pd.read_excel(os.path.join(BASE_DIR, 'City Council District 3 Data.xlsx'))
gdf = gpd.read_file(os.path.join(BASE_DIR, 'tl_2020_01_vtd20.shp'))

# Build GEOID20 to match shapefile format: state (01) + county (089) + precinct (padded to 6 digits)
df['GEOID20_full'] = '01089' + df['GEOID20'].astype(str).str.zfill(6)

# Convert shapefile GEOID20 to string for matching
gdf['GEOID20'] = gdf['GEOID20'].astype(str)

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

# Function to prepare data for a specific year
def prepare_year_data(year):
    merged_temp = gdf.merge(df, left_on='GEOID20', right_on='GEOID20_full', how='inner')
    
    if year == 2018:
        # Rename columns to standardize candidate names
        merged_temp = merged_temp.rename(columns={'Robinson 2018': 'Cand1', 'Iley 2018': 'Cand2', 'Schexnayder': 'Cand3'})
        candidates = ['Cand1', 'Cand2', 'Cand3']
        cand_names = {'Cand1': 'Robinson', 'Cand2': 'Iley', 'Cand3': 'Schexnayder'}
    else:  # 2014
        merged_temp = merged_temp.rename(columns={'Robinson 2014': 'Cand1', 'Hennessee 2014': 'Cand2', 'Shexnayder 2014': 'Cand3'})
        candidates = ['Cand1', 'Cand2', 'Cand3']
        cand_names = {'Cand1': 'Robinson', 'Cand2': 'Hennessee', 'Cand3': 'Shexnayder'}
    
    merged_temp['Total_Votes'] = merged_temp[candidates].sum(axis=1)
    
    for cand in candidates:
        merged_temp[f'{cand}_pct'] = (merged_temp[cand] / merged_temp['Total_Votes'] * 100).round(1)
    
    # Pairwise comparisons
    merged_temp['Cand1_vs_Cand2'] = merged_temp['Cand1'] - merged_temp['Cand2']
    merged_temp['Cand1_vs_Cand3'] = merged_temp['Cand1'] - merged_temp['Cand3']
    merged_temp['Cand2_vs_Cand3'] = merged_temp['Cand2'] - merged_temp['Cand3']
    
    merged_temp['Cand1_vs_Cand2_pct'] = ((merged_temp['Cand1'] - merged_temp['Cand2']) / merged_temp['Total_Votes'] * 100).round(1)
    merged_temp['Cand1_vs_Cand3_pct'] = ((merged_temp['Cand1'] - merged_temp['Cand3']) / merged_temp['Total_Votes'] * 100).round(1)
    merged_temp['Cand2_vs_Cand3_pct'] = ((merged_temp['Cand2'] - merged_temp['Cand3']) / merged_temp['Total_Votes'] * 100).round(1)
    
    # Winner calculation
    def get_winner(row):
        votes = {cand_names['Cand1']: row['Cand1'], cand_names['Cand2']: row['Cand2'], cand_names['Cand3']: row['Cand3']}
        return max(votes, key=votes.get)
    
    def get_winner_margin(row):
        votes = sorted([row['Cand1'], row['Cand2'], row['Cand3']], reverse=True)
        return votes[0] - votes[1]
    
    merged_temp['Winner'] = merged_temp.apply(get_winner, axis=1)
    merged_temp['Winner_Margin'] = merged_temp.apply(get_winner_margin, axis=1)
    merged_temp['Winner_Margin_Pct'] = (merged_temp['Winner_Margin'] / merged_temp['Total_Votes'] * 100).round(1)
    
    # Assign numeric codes for coloring
    winner_map = {cand_names['Cand1']: 0, cand_names['Cand2']: 1, cand_names['Cand3']: 2}
    merged_temp['Winner_Code'] = merged_temp['Winner'].map(winner_map)
    
    # Convert to WGS84 for mapping
    merged_temp = merged_temp.to_crs(epsg=4326)
    
    # Calculate centroids for bubble overlay
    merged_projected = merged_temp.to_crs(epsg=3857)
    centroids = merged_projected.geometry.centroid.to_crs(epsg=4326)
    merged_temp['centroid_lat'] = centroids.y
    merged_temp['centroid_lon'] = centroids.x
    
    # Normalize bubble sizes
    max_votes = merged_temp['Total_Votes'].max()
    min_votes = merged_temp['Total_Votes'].min()
    merged_temp['bubble_size'] = 10 + 40 * (merged_temp['Total_Votes'] - min_votes) / (max_votes - min_votes) if max_votes > min_votes else 25
    
    # Calculate winner colors by margin
    cand_colors = {cand_names['Cand1']: '#000080', cand_names['Cand2']: '#FFD700', cand_names['Cand3']: '#FF00FF'}
    max_margin = merged_temp['Winner_Margin'].max()
    max_margin_pct = merged_temp['Winner_Margin_Pct'].max()
    
    def get_winner_color_by_margin(row, use_pct=False):
        winner = row['Winner']
        base_color = cand_colors[winner]
        if use_pct:
            intensity = min(row['Winner_Margin_Pct'] / max_margin_pct, 1.0) if max_margin_pct > 0 else 0.5
        else:
            intensity = min(row['Winner_Margin'] / max_margin, 1.0) if max_margin > 0 else 0.5
        intensity = 0.3 + 0.7 * intensity
        return blend_to_white(base_color, intensity)
    
    merged_temp['Winner_Color_Margin'] = merged_temp.apply(lambda r: get_winner_color_by_margin(r, False), axis=1)
    merged_temp['Winner_Color_Margin_Pct'] = merged_temp.apply(lambda r: get_winner_color_by_margin(r, True), axis=1)
    
    return merged_temp, cand_names

# Initialize with 2018 data
merged, candidate_labels = prepare_year_data(2018)
candidates = ['Cand1', 'Cand2', 'Cand3']

# Get GeoJSON for Plotly
geojson = json.loads(merged.to_json())

# Map center calculation
center_lat = merged['centroid_lat'].mean()
center_lon = merged['centroid_lon'].mean()

# Candidate colors - Navy, Gold, Magenta for better contrast (will be updated dynamically)
candidate_colors = {candidate_labels['Cand1']: '#000080', candidate_labels['Cand2']: '#FFD700', candidate_labels['Cand3']: '#FF00FF'}

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("City Council District 3 - Election Results", className="text-center mb-4 mt-3", id='dashboard-title')
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("View Options"),
                dbc.CardBody([
                    html.Label("Select Election Year:", className="fw-bold"),
                    dcc.RadioItems(
                        id='year-select',
                        options=[
                            {'label': ' 2014 Election', 'value': 2014},
                            {'label': ' 2018 Election', 'value': 2018}
                        ],
                        value=2018,
                        className="mb-3"
                    ),
                    html.Hr(),
                    html.Label("Select View:", className="fw-bold"),
                    dcc.RadioItems(
                        id='view-type',
                        options=[
                            {'label': ' Individual Candidate', 'value': 'individual'},
                            {'label': ' Head-to-Head (2-way)', 'value': 'comparison'},
                            {'label': ' Three-Way Comparison', 'value': 'threeway'},
                            {'label': ' Turnout Change (2014 vs 2018)', 'value': 'turnout_change'}
                        ],
                        value='individual',
                        className="mb-3"
                    ),
                    # Candidate selector (shown for individual view)
                    html.Div(id='candidate-container', children=[
                        html.Label("Select Candidate:", className="fw-bold mt-2"),
                        dcc.Dropdown(
                            id='candidate-select',
                            options=[],
                            value='Cand1',
                            clearable=False
                        )
                    ]),
                    # Comparison selector (shown for head-to-head view)
                    html.Div(id='comparison-container', style={'display': 'none'}, children=[
                        html.Label("Select Comparison:", className="fw-bold mt-2"),
                        dcc.Dropdown(
                            id='comparison-select',
                            options=[],
                            value='Cand1_vs_Cand2',
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
    [Output('candidate-select', 'options'),
     Output('candidate-select', 'value'),
     Output('comparison-select', 'options'),
     Output('comparison-select', 'value'),
     Output('dashboard-title', 'children')],
    [Input('year-select', 'value')],
    prevent_initial_call=False
)
def update_year(year):
    global merged, candidate_labels, candidate_colors, candidates, geojson, center_lat, center_lon
    merged, candidate_labels = prepare_year_data(year)
    candidates = ['Cand1', 'Cand2', 'Cand3']
    candidate_colors = {candidate_labels['Cand1']: '#000080', candidate_labels['Cand2']: '#FFD700', candidate_labels['Cand3']: '#FF00FF'}
    
    # Regenerate GeoJSON and map center
    geojson = json.loads(merged.to_json())
    center_lat = merged['centroid_lat'].mean()
    center_lon = merged['centroid_lon'].mean()
    
    cand_options = [
        {'label': candidate_labels['Cand1'], 'value': 'Cand1'},
        {'label': candidate_labels['Cand2'], 'value': 'Cand2'},
        {'label': candidate_labels['Cand3'], 'value': 'Cand3'}
    ]
    
    comp_options = [
        {'label': f"{candidate_labels['Cand1']} vs {candidate_labels['Cand2']}", 'value': 'Cand1_vs_Cand2'},
        {'label': f"{candidate_labels['Cand1']} vs {candidate_labels['Cand3']}", 'value': 'Cand1_vs_Cand3'},
        {'label': f"{candidate_labels['Cand2']} vs {candidate_labels['Cand3']}", 'value': 'Cand2_vs_Cand3'}
    ]
    
    title = f"City Council District 3 - {year} Election Results"
    
    # Return Cand1 (Robinson) as default
    return cand_options, 'Cand1', comp_options, 'Cand1_vs_Cand2', title


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
    elif view_type == 'threeway':
        metric_options = [
            {'label': ' Winner margin (votes)', 'value': 'margin_count'},
            {'label': ' Winner margin (%)', 'value': 'margin_pct'}
        ]
        return ({'display': 'none'}, {'display': 'none'}, metric_options, 'margin_count')
    else:  # turnout_change
        metric_options = [
            {'label': ' Vote Change', 'value': 'vote_change'},
            {'label': ' Percent Change', 'value': 'pct_change'}
        ]
        return ({'display': 'none'}, {'display': 'none'}, metric_options, 'vote_change')


@callback(
    Output('legend-container', 'children'),
    [Input('view-type', 'value'),
     Input('year-select', 'value')]
)
def update_legend(view_type, year):
    if view_type == 'turnout_change':
        return html.Div([
            html.Div([
                html.Span("■ ", style={'color': '#DC143C', 'fontSize': '20px'}),
                html.Span("Decreased Turnout")
            ]),
            html.Div([
                html.Span("■ ", style={'color': '#FFFFFF', 'fontSize': '20px', 'textShadow': '0 0 1px #000'}),
                html.Span("No Change")
            ]),
            html.Div([
                html.Span("■ ", style={'color': '#1E90FF', 'fontSize': '20px'}),
                html.Span("Increased Turnout")
            ]),
            html.Hr(),
            html.Small("Comparing total votes: 2018 vs 2014", className="text-muted")
        ])
    elif view_type == 'threeway':
        return html.Div([
            html.Div([
                html.Span("■ ", style={'color': candidate_colors[candidate_labels['Cand1']], 'fontSize': '20px'}),
                html.Span(candidate_labels['Cand1'])
            ]),
            html.Div([
                html.Span("■ ", style={'color': candidate_colors[candidate_labels['Cand2']], 'fontSize': '20px'}),
                html.Span(candidate_labels['Cand2'])
            ]),
            html.Div([
                html.Span("■ ", style={'color': candidate_colors[candidate_labels['Cand3']], 'fontSize': '20px'}),
                html.Span(candidate_labels['Cand3'])
            ]),
            html.Hr(),
            html.Small("Colors show which candidate won each precinct", className="text-muted")
        ])
    elif view_type == 'comparison':
        return html.Div([
            html.Div([
                html.Span("■ ", style={'color': candidate_colors[candidate_labels['Cand1']], 'fontSize': '20px'}),
                html.Span(f"{candidate_labels['Cand1']} (Navy) - positive margin")
            ]),
            html.Div([
                html.Span("■ ", style={'color': candidate_colors[candidate_labels['Cand2']], 'fontSize': '20px'}),
                html.Span(f"{candidate_labels['Cand2']} (Gold)")
            ]),
            html.Div([
                html.Span("■ ", style={'color': candidate_colors[candidate_labels['Cand3']], 'fontSize': '20px'}),
                html.Span(f"{candidate_labels['Cand3']} (Magenta)")
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
     Input('show-bubbles', 'value'),
     Input('year-select', 'value')]
)
def update_map(view_type, candidate, comparison, metric_type, show_bubbles, year):
    print(f"update_map called: view_type={view_type}, metric_type={metric_type}, year={year}")
    show_volume = 'show' in show_bubbles if show_bubbles else False
    
    if view_type == 'individual':
        if candidate is None or candidate not in ['Cand1', 'Cand2', 'Cand3']:
            candidate = 'Cand1'
        
        # Handle invalid metric_type when switching from threeway
        if metric_type not in ['percent', 'count']:
            metric_type = 'percent'
        
        cand_label = candidate_labels[candidate]
        
        if metric_type == 'percent':
            color_col = f'{candidate}_pct'
            title = f'{cand_label} - Vote Share (%)'
            color_label = 'Vote %'
        else:
            color_col = candidate
            title = f'{cand_label} - Vote Count'
            color_label = 'Votes'
        
        # Create color scale from white to candidate's color
        cand_color = candidate_colors[cand_label]
        individual_scale = [[0, '#FFFFFF'], [1, cand_color]]
        
        # Set explicit range for color scale
        if metric_type == 'percent':
            range_color = [0, 100]
        else:
            range_color = [merged[color_col].min(), merged[color_col].max()]
        
        fig = px.choropleth_map(
            merged,
            geojson=geojson,
            locations=merged.index,
            color=color_col,
            hover_name='Precinct Name',
            hover_data={
                'Cand1': True,
                'Cand2': True,
                'Cand3': True,
                'Total_Votes': True,
                'Cand1_pct': ':.1f',
                'Cand2_pct': ':.1f',
                'Cand3_pct': ':.1f'
            },
            labels={
                'Cand1': candidate_labels['Cand1'],
                'Cand2': candidate_labels['Cand2'],
                'Cand3': candidate_labels['Cand3'],
                color_col: color_label
            },
            color_continuous_scale=individual_scale,
            range_color=range_color,
            map_style='carto-positron',
            zoom=10,
            center={'lat': center_lat, 'lon': center_lon},
            opacity=0.7
        )
        
    elif view_type == 'comparison':
        if comparison is None or comparison not in ['Cand1_vs_Cand2', 'Cand1_vs_Cand3', 'Cand2_vs_Cand3']:
            comparison = 'Cand1_vs_Cand2'
        
        # Handle invalid metric_type when switching from threeway
        if metric_type not in ['percent', 'count']:
            metric_type = 'percent'
        
        cand1, cand2 = comparison.split('_vs_')
        cand1_label = candidate_labels[cand1]
        cand2_label = candidate_labels[cand2]
        
        if metric_type == 'percent':
            color_col = f'{comparison}_pct'
            title = f'{cand1_label} vs {cand2_label} - Margin (%)'
            color_label = 'Margin %'
        else:
            color_col = comparison
            title = f'{cand1_label} vs {cand2_label} - Vote Margin'
            color_label = 'Vote Margin'
        
        # Create custom color scale using candidate colors
        color1 = candidate_colors[cand1_label]
        color2 = candidate_colors[cand2_label]
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
            labels={
                cand1: cand1_label,
                cand2: cand2_label,
                color_col: color_label
            },
            color_continuous_scale=custom_scale,
            color_continuous_midpoint=0,
            map_style='carto-positron',
            zoom=10,
            center={'lat': center_lat, 'lon': center_lon},
            opacity=0.7
        )
        
    elif view_type == 'threeway':
        if metric_type == 'margin_count':
            # Use Winner_Color_Margin for intensity-based coloring
            fig = px.choropleth_map(
                merged,
                geojson=geojson,
                locations=merged.index,
                color='Winner_Color_Margin',
                hover_name='Precinct Name',
                hover_data={
                    'Winner': True,
                    'Cand1': True,
                    'Cand2': True,
                    'Cand3': True,
                    'Total_Votes': True,
                    'Winner_Margin': True,
                    'Winner_Margin_Pct': ':.1f'
                },
                labels={
                    'Cand1': candidate_labels['Cand1'],
                    'Cand2': candidate_labels['Cand2'],
                    'Cand3': candidate_labels['Cand3']
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
                    'Cand1': True,
                    'Cand2': True,
                    'Cand3': True,
                    'Total_Votes': True,
                    'Winner_Margin': True,
                    'Winner_Margin_Pct': ':.1f'
                },
                labels={
                    'Cand1': candidate_labels['Cand1'],
                    'Cand2': candidate_labels['Cand2'],
                    'Cand3': candidate_labels['Cand3']
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
            # Fallback to margin_count if metric_type is unexpected
            metric_type = 'margin_count'
            fig = px.choropleth_map(
                merged,
                geojson=geojson,
                locations=merged.index,
                color='Winner_Color_Margin',
                hover_name='Precinct Name',
                hover_data={
                    'Winner': True,
                    'Cand1': True,
                    'Cand2': True,
                    'Cand3': True,
                    'Total_Votes': True,
                    'Winner_Margin': True,
                    'Winner_Margin_Pct': ':.1f'
                },
                labels={
                    'Cand1': candidate_labels['Cand1'],
                    'Cand2': candidate_labels['Cand2'],
                    'Cand3': candidate_labels['Cand3']
                },
                color_discrete_map={c: c for c in merged['Winner_Color_Margin'].unique()},
                map_style='carto-positron',
                zoom=10,
                center={'lat': center_lat, 'lon': center_lon},
                opacity=0.85
            )
            fig.update_layout(showlegend=False)
            title = 'Three-Way: Winner Margin (votes) - Intensity = Margin Size'
    
    elif view_type == 'turnout_change':
        # Load both years and compare turnout
        merged_2014, labels_2014 = prepare_year_data(2014)
        merged_2018, labels_2018 = prepare_year_data(2018)
        
        # Merge on precinct name to compare
        comparison_df = merged_2018[['Precinct Name', 'Total_Votes', 'geometry']].copy()
        comparison_df = comparison_df.rename(columns={'Total_Votes': 'Votes_2018'})
        
        votes_2014 = merged_2014[['Precinct Name', 'Total_Votes']].copy()
        votes_2014 = votes_2014.rename(columns={'Total_Votes': 'Votes_2014'})
        
        comparison_df = comparison_df.merge(votes_2014, on='Precinct Name', how='left')
        
        # Calculate changes
        comparison_df['Vote_Change'] = comparison_df['Votes_2018'] - comparison_df['Votes_2014']
        comparison_df['Pct_Change'] = ((comparison_df['Votes_2018'] - comparison_df['Votes_2014']) / comparison_df['Votes_2014'] * 100).round(1)
        
        # Create GeoDataFrame for plotting
        comparison_gdf = gpd.GeoDataFrame(comparison_df, geometry='geometry')
        comparison_geojson = json.loads(comparison_gdf.to_json())
        
        if metric_type == 'vote_change':
            color_col = 'Vote_Change'
            color_label = 'Vote Change'
            title = 'Turnout Change: Total Votes (2018 - 2014)'
        else:
            color_col = 'Pct_Change'
            color_label = '% Change'
            title = 'Turnout Change: Percent (2018 - 2014)'
        
        # Diverging color scale: red (negative) -> white (0) -> blue (positive)
        fig = px.choropleth_map(
            comparison_gdf,
            geojson=comparison_geojson,
            locations=comparison_gdf.index,
            color=color_col,
            hover_name='Precinct Name',
            hover_data={
                'Votes_2014': True,
                'Votes_2018': True,
                'Vote_Change': True,
                'Pct_Change': ':.1f'
            },
            labels={
                'Votes_2014': '2014 Total',
                'Votes_2018': '2018 Total',
                'Vote_Change': 'Change',
                'Pct_Change': '% Change'
            },
            color_continuous_scale=['#DC143C', '#FFFFFF', '#1E90FF'],  # Red -> White -> Blue
            color_continuous_midpoint=0,
            map_style='carto-positron',
            zoom=10,
            center={'lat': center_lat, 'lon': center_lon},
            opacity=0.7
        )
    
    # Add pie chart overlay for vote volume if enabled
    if show_volume:
        import math
        # Add pie charts as polygon traces for each precinct
        for idx, row in merged.iterrows():
            total = row['Total_Votes']
            if total == 0:
                continue
                
            cand1_pct = row['Cand1'] / total
            cand2_pct = row['Cand2'] / total
            cand3_pct = row['Cand3'] / total
            
            # Scale size based on total votes - smaller radius for better fit
            r = 0.002 + 0.004 * (row['bubble_size'] - 10) / 40  # radius in degrees
            
            lat_center = row['centroid_lat']
            lon_center = row['centroid_lon']
            
            hover_text = (f"<b>{row['Precinct Name']}</b><br>"
                         f"Total: {total}<br>"
                         f"{candidate_labels['Cand1']}: {row['Cand1']} ({row['Cand1_pct']}%)<br>"
                         f"{candidate_labels['Cand2']}: {row['Cand2']} ({row['Cand2_pct']}%)<br>"
                         f"{candidate_labels['Cand3']}: {row['Cand3']} ({row['Cand3_pct']}%)")
            
            n_points = 20  # points per slice for smooth edges
            pcts = [cand1_pct, cand2_pct, cand3_pct]
            pie_colors = [candidate_colors[candidate_labels['Cand1']], candidate_colors[candidate_labels['Cand2']], candidate_colors[candidate_labels['Cand3']]]
            labels = [candidate_labels['Cand1'], candidate_labels['Cand2'], candidate_labels['Cand3']]
            
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
     Input('comparison-select', 'value'),
     Input('year-select', 'value')]
)
def update_table(view_type, candidate, comparison, year):
    if view_type == 'turnout_change':
        # Show comparison table
        merged_2014, labels_2014 = prepare_year_data(2014)
        merged_2018, labels_2018 = prepare_year_data(2018)
        
        comparison_df = merged_2018[['Precinct Name', 'Total_Votes']].copy()
        comparison_df = comparison_df.rename(columns={'Total_Votes': 'Votes_2018'})
        
        votes_2014 = merged_2014[['Precinct Name', 'Total_Votes']].copy()
        votes_2014 = votes_2014.rename(columns={'Total_Votes': 'Votes_2014'})
        
        comparison_df = comparison_df.merge(votes_2014, on='Precinct Name', how='left')
        comparison_df['Change'] = comparison_df['Votes_2018'] - comparison_df['Votes_2014']
        comparison_df['% Change'] = ((comparison_df['Votes_2018'] - comparison_df['Votes_2014']) / comparison_df['Votes_2014'] * 100).round(1)
        
        table_df = comparison_df[['Precinct Name', 'Votes_2014', 'Votes_2018', 'Change', '% Change']]
        table_df.columns = ['Precinct', '2014 Total', '2018 Total', 'Change', '% Change']
        table_df = table_df.sort_values('Change', ascending=False)
    elif view_type == 'threeway':
        table_df = merged[['Precinct Name', 'Cand1', 'Cand2', 'Cand3', 'Total_Votes', 
                           'Winner', 'Winner_Margin', 'Winner_Margin_Pct']].copy()
        table_df.columns = ['Precinct', candidate_labels['Cand1'], candidate_labels['Cand2'], candidate_labels['Cand3'], 'Total', 
                            'Winner', 'Margin', 'Margin %']
        table_df = table_df.sort_values('Total', ascending=False)
    else:
        table_df = merged[['Precinct Name', 'Cand1', 'Cand2', 'Cand3', 'Total_Votes', 
                           'Cand1_pct', 'Cand2_pct', 'Cand3_pct']].copy()
        table_df.columns = ['Precinct', candidate_labels['Cand1'], candidate_labels['Cand2'], candidate_labels['Cand3'], 'Total', 
                            f"{candidate_labels['Cand1']} %", f"{candidate_labels['Cand2']} %", f"{candidate_labels['Cand3']} %"]
        
        if view_type == 'individual' and candidate:
            table_df = table_df.sort_values(f'{candidate_labels[candidate]} %', ascending=False)
        elif view_type == 'comparison' and comparison:
            cand1, cand2 = comparison.split('_vs_')
            table_df = table_df.sort_values(candidate_labels[cand1], ascending=False)
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
    [Input('view-type', 'value'),
     Input('year-select', 'value')]
)
def update_bar(view_type, year):
    # Regenerate data for the selected year to ensure bar chart shows correct totals
    current_merged, current_labels = prepare_year_data(year)
    current_colors_dict = {current_labels['Cand1']: '#000080', current_labels['Cand2']: '#FFD700', current_labels['Cand3']: '#FF00FF'}
    
    totals = {
        current_labels['Cand1']: current_merged['Cand1'].sum(),
        current_labels['Cand2']: current_merged['Cand2'].sum(),
        current_labels['Cand3']: current_merged['Cand3'].sum()
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(totals.keys()),
            y=list(totals.values()),
            marker_color=[current_colors_dict[current_labels['Cand1']], current_colors_dict[current_labels['Cand2']], current_colors_dict[current_labels['Cand3']]],
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
    print(f"Total votes - {candidate_labels['Cand1']}: {merged['Cand1'].sum()}, {candidate_labels['Cand2']}: {merged['Cand2'].sum()}, {candidate_labels['Cand3']}: {merged['Cand3'].sum()}")
    print(f"\nPrecincts won by each candidate:")
    print(merged['Winner'].value_counts().to_string())
    print("\nStarting dashboard server...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True)
