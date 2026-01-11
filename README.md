# Huntsville City Council District 3 Election Dashboard

An interactive dashboard visualizing election results for Huntsville City Council District 3, comparing the 2014 and 2018 elections.

## Features

- **Year Comparison**: Switch between 2014 and 2018 election data
- **Multiple View Types**:
  - Individual candidate performance
  - Head-to-head comparisons
  - Three-way winner visualization with margins
- **Interactive Maps**: Click and hover over precincts to see detailed results
- **Vote Volume Overlays**: Pie chart visualization showing vote distribution
- **Geospatial Analysis**: Precinct-level breakdown with vote percentages

## Candidates

### 2018 Election
- Robinson
- Iley
- Schexnayder

### 2014 Election
- Robinson
- Hennessee
- Shexnayder

## Technology Stack

- **Python** - Core language
- **Dash** - Interactive web framework
- **Plotly** - Data visualization
- **Geopandas** - Geospatial data processing
- **Pandas** - Data manipulation

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the dashboard:
   ```bash
   python dashboard.py
   ```

3. Open your browser to `http://127.0.0.1:8050`

## Deployment

This dashboard is configured for deployment on Render.com using the included `render.yaml` and `Procfile`.

## Data Sources

- Election data: City Council District 3 Data.xlsx
- Geospatial data: U.S. Census Bureau Voting District shapefiles (2020)
