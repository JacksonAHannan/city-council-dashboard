import pandas as pd
import geopandas as gpd

# Read Excel file
df = pd.read_excel(r'c:\Users\User\Desktop\City Council 2 Project\City Council District 2 2022 Data.xlsx')
print('Excel Data Columns:', df.columns.tolist())
print('\nShape:', df.shape)
print('\nFirst 15 rows:\n', df.head(15))
print('\nData types:\n', df.dtypes)

# Check for shapefile components
import os
base_path = r'c:\Users\User\Desktop\City Council 2 Project\tl_2020_01_vtd20'
extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
print('\n\nShapefile components:')
for ext in extensions:
    path = base_path + ext
    exists = os.path.exists(path)
    print(f'{ext}: {"EXISTS" if exists else "MISSING"}')

# Try to read shapefile
try:
    gdf = gpd.read_file(r'c:\Users\User\Desktop\City Council 2 Project\tl_2020_01_vtd20.shp')
    print('\n\nShapefile Columns:', gdf.columns.tolist())
    print('\nShapefile Shape:', gdf.shape)
    print('\nFirst 5 rows (no geometry):\n', gdf.drop(columns='geometry').head())
except Exception as e:
    print(f'\nError reading shapefile: {e}')
