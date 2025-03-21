# Climate Trend Analysis

## Project Overview
This project analyzes climate trends in Nepal, comparing temperature and precipitation data between 2020 and 2050. Using GIS data, raster analysis, and statistical visualizations, the project aims to highlight future climate changes and their potential impacts.

## Data Sources
- **Vector Data**: Nepal's district boundaries from `Shape_Data/local_unit.shp`
- **Raster Data**: Climate data in GeoTIFF format:
  - `nepal_climate_data/nepal_temperature_2020.tif`
  - `nepal_climate_data/nepal_temperature_2050.tif`
  - `nepal_climate_data/nepal_precipitation_2020.tif`
  - `nepal_climate_data/nepal_precipitation_2050.tif`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/climate-trend-analysis.git
   cd climate-trend-analysis
   ```
2. Install required Python packages:
   ```bash
   pip install geopandas rasterio numpy pandas matplotlib seaborn
   ```

## Running the Project
1. Ensure vector and raster datasets are placed in the appropriate directories.
2. Run the script:
   ```bash
   python climate_trend_analysis.py
   ```
3. The script performs the following:
   - Loads vector and raster data.
   - Overlays district boundaries on climate data.
   - Visualizes trends using bar graphs and line charts.
   - Prints summary statistics for temperature and precipitation.

## Key Findings
- **Temperature Trends**: Mean temperatures show a clear upward trend from 2020 to 2050.
- **Precipitation Trends**: Changes in precipitation are more variable, with slight increases observed.
- **Visualizations**: Maps, bar charts, and line graphs reveal district-level variations and long-term trends.

## Results
The project produces:
- Climate maps for 2020 and 2050.
- Statistical comparisons of temperature and precipitation.
- Line graphs illustrating long-term climate changes.


