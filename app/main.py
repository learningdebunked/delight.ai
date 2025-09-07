import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import json

# Initialize the Dash app
app = Dash(__name__, 
          external_stylesheets=[dbc.themes.BOOTSTRAP],
          suppress_callback_exceptions=True)

# Sample data loading function (replace with your actual data loading)
def load_data():
    try:
        # Try to load the generated data
        df = pd.read_csv('../data/raw/service_interactions.csv')
        # Convert string representation of dict to actual dict for cultural_profile and emotion_state
        if 'cultural_profile' in df.columns and isinstance(df['cultural_profile'].iloc[0], str):
            df['cultural_profile'] = df['cultural_profile'].apply(eval)
        if 'emotion_state' in df.columns and isinstance(df['emotion_state'].iloc[0], str):
            df['emotion_state'] = df['emotion_state'].apply(eval)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return sample data if file not found
        return pd.DataFrame({
            'interaction_id': range(10),
            'timestamp': [datetime.now() - timedelta(days=x) for x in range(10)],
            'region': ['north_america']*5 + ['europe']*5,
            'scenario': ['product_inquiry']*3 + ['complaint']*3 + ['technical_support']*4,
            'satisfaction_score': np.random.uniform(0.3, 1.0, 10),
            'resolution_status': ['resolved']*7 + ['escalated']*3
        })

# Load the data
df = load_data()

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("SEDS Framework Dashboard", className="text-center my-4"), width=12)
    ]),
    
    # Tabs for different views
    dbc.Tabs([
        # Overview Tab
        dbc.Tab(label="Overview", children=[
            dbc.Row([
                # Key Metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Interactions", className="card-title"),
                            html.H2(f"{len(df):,}", className="card-text text-center")
                        ])
                    ], className="mb-4"),
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Avg. Satisfaction", className="card-title"),
                            html.H2(f"{df['satisfaction_score'].mean():.1%}", 
                                  className="card-text text-center")
                        ])
                    ])
                ], md=3),
                
                # Satisfaction Over Time
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Satisfaction Over Time", className="card-title"),
                            dcc.Graph(id='satisfaction-trend')
                        ])
                    ])
                ], md=9)
            ], className="mb-4"),
            
            dbc.Row([
                # Region Distribution
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Interactions by Region", className="card-title"),
                            dcc.Graph(id='region-dist')
                        ])
                    ])
                ], md=6),
                
                # Scenario Distribution
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Interactions by Scenario", className="card-title"),
                            dcc.Graph(id='scenario-dist')
                        ])
                    ])
                ], md=6)
            ])
        ]),
        
        # Cultural Analysis Tab
        dbc.Tab(label="Cultural Analysis", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Cultural Dimension Analysis", className="card-title"),
                            dcc.Dropdown(
                                id='dimension-selector',
                                options=[{'label': dim, 'value': dim} 
                                        for dim in ['individualism', 'power_distance', 'uncertainty_avoidance', 'long_term_orientation']],
                                value='individualism',
                                className="mb-3"
                            ),
                            dcc.Graph(id='cultural-heatmap')
                        ])
                    ])
                ], width=12)
            ])
        ]),
        
        # Emotion Analysis Tab
        dbc.Tab(label="Emotion Analysis", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Emotion Distribution by Scenario", className="card-title"),
                            dcc.Graph(id='emotion-dist')
                        ])
                    ])
                ], width=12)
            ])
        ]),
        
        # Raw Data Tab
        dbc.Tab(label="Raw Data", children=[
            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(
                        id='data-table',
                        columns=[{"name": i, "id": i} for i in df.columns if i not in ['cultural_profile', 'emotion_state']],
                        data=df.to_dict('records'),
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'height': 'auto',
                            'minWidth': '100px', 'maxWidth': '200px',
                            'whiteSpace': 'normal',
                            'textAlign': 'left'
                        },
                        filter_action="native",
                        sort_action="native"
                    )
                ], width=12)
            ])
        ])
    ])
], fluid=True)

# Callbacks for interactivity
@app.callback(
    Output('satisfaction-trend', 'figure'),
    [Input('data-table', 'data')]
)
def update_satisfaction_trend(data):
    if not data:
        return go.Figure()
    
    df_trend = pd.DataFrame(data)
    df_trend['timestamp'] = pd.to_datetime(df_trend['timestamp'])
    df_trend = df_trend.set_index('timestamp')
    
    # Resample to daily average
    df_daily = df_trend['satisfaction_score'].resample('D').mean().reset_index()
    
    fig = px.line(df_daily, x='timestamp', y='satisfaction_score',
                 title='Daily Average Satisfaction Score',
                 labels={'satisfaction_score': 'Satisfaction Score', 'timestamp': 'Date'})
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Satisfaction Score',
        yaxis_tickformat='.0%',
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    Output('region-dist', 'figure'),
    [Input('data-table', 'data')]
)
def update_region_dist(data):
    if not data:
        return go.Figure()
    
    df_region = pd.DataFrame(data)
    region_counts = df_region['region'].value_counts().reset_index()
    region_counts.columns = ['region', 'count']
    
    fig = px.pie(region_counts, values='count', names='region',
                title='Interactions by Region',
                hole=0.3)
    
    return fig

@app.callback(
    Output('scenario-dist', 'figure'),
    [Input('data-table', 'data')]
)
def update_scenario_dist(data):
    if not data:
        return go.Figure()
    
    df_scenario = pd.DataFrame(data)
    scenario_counts = df_scenario['scenario'].value_counts().reset_index()
    scenario_counts.columns = ['scenario', 'count']
    
    fig = px.bar(scenario_counts, x='scenario', y='count',
                title='Interactions by Scenario',
                labels={'scenario': 'Scenario', 'count': 'Count'})
    
    fig.update_layout(xaxis_tickangle=-45)
    return fig

@app.callback(
    Output('emotion-dist', 'figure'),
    [Input('data-table', 'data')]
)
def update_emotion_dist(data):
    if not data:
        return go.Figure()
    
    df_emotion = pd.DataFrame(data)
    
    # Extract emotions from the emotion_state column
    emotions = []
    for _, row in df_emotion.iterrows():
        if isinstance(row.get('emotion_state'), dict):
            for emotion, score in row['emotion_state'].items():
                emotions.append({
                    'scenario': row['scenario'],
                    'emotion': emotion,
                    'score': score
                })
    
    if not emotions:
        return go.Figure()
        
    df_emotions = pd.DataFrame(emotions)
    
    # Calculate average emotion scores by scenario
    emotion_means = df_emotions.groupby(['scenario', 'emotion'])['score'].mean().reset_index()
    
    fig = px.bar(emotion_means, x='scenario', y='score', color='emotion',
                title='Average Emotion Intensity by Scenario',
                labels={'scenario': 'Scenario', 'score': 'Average Intensity'},
                barmode='group')
    
    fig.update_layout(xaxis_tickangle=-45)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
