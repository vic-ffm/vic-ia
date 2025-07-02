# Data

Contains version 2.5.8 of the Victorian Wildfire Research Dataset, joined with predictors, in `vwfrd_v2.5.8.csv`. Each row represents a fire that occurred within the state of Victoria, Australia between 1 April 2008 and 30 March 2024.


To reference this data:

Butler K, Rennie J, Tartaglia E (2025). 'Victorian Wildfire Research Dataset v2.5.8 joined with predictors' (v1) [Dataset]. Country Fire Authority and Department of Energy, Environment and Climate Action, Melbourne, Australia. 

| Variable | Name | Units | Description |
| -------- | ---- | ----- | ----------- |
| `season` | Fire season | - | Range of years indicating the fire season, e.g. 2008-09 indicates the fire season over the Australian summer of 2008 to 2009. 
| `date` | Date | - | Date that the fire was reported to have started. Only recorded for fires in the case studies for privacy.
| `reported_time` | Reported time | - | Date and time the fire was reported to have started. Only recorded for fires in the case studies for privacy.
| `T_SFC` | Temperature | &deg;C | Ambient air temperature from the most recent forecast before the fire report time from the Australian Digital Forecast Data, Bureau of Meteorology.
| `T_SFC_historical` | Temperature (historical) | &deg;C | Ambient air temperature from the VicClim grid value closest to the reported fire time and location.
| `RH_SFC` | Relative humidity | % |  Relative humidity as a percentage from the most recent forecast before the fire report time from the Australian Digital Forecast Data (Bureau of Meteorology).
| `RH_SFC_historical` | Relative humidity (historical) | % |  Relative humidity as a percentage from the VicClim grid value closest to the reported fire time and location.
| `WindMagKmh_SFC` | Wind speed | km/h |  Wind speed from the most recent forecast before the fire report time from the Australian Digital Forecast Data, Bureau of Meteorology.
| `WindMagKmh_SFC_historical` | Wind speed (historical) | km/h |  Wind speed from the VicClim grid value closest to the reported fire time and location.
| `DF_SFC` | Drought factor | - |  Drought factor from the most recent forecast before the fire report time from the Australian Digital Forecast Data, Bureau of Meteorology.
| `DF_SFC_historical` | Drought factor (historical) | - |  Drought factor from the VicClim grid value closest to the reported fire time and location.
| `KBDI_SFC` | KBDI | - |  Keetch Byram Drought Index from the most recent forecast before the fire report time from the Australian Digital Forecast Data, Bureau of Meteorology.
| `KBDI_SFC_historical` | KBDI (historical) | - | Keetch Byram Drought Index from the VicClim grid value closest to the reported fire time and location.
| `soil_moisture` | Soil moisture content | - | Soil moisture content (root) from the closest time and location in the Australian Water Outlook, Bureau of Meteorology.
| `Curing` | Curing | - |  A measure of grass dryness from the most recent forecast before the fire report time from the Australian Digital Forecast Data, Bureau of Meteorology. The forecast is value comes from the last observed value of curing.
| `Curing_historical` | Curing (historical) | - | A measure of grass dryness from the VicClim grid value closest to the reported fire time and location. The curing values in VicClim were developed by the Country Fire Authority.
| `grass_density_3km` | Grass density (3 km) | - | Fraction of (30m by 30 m) grid cells that are categorised as grass by a Forest Fire Management Victoria fuel layer in a grid with the fire start location at the centre and apothem 3 km.
| `forest_density_3km` | Forest density (3 km) | - | Fraction of (30m by 30 m) grid cells that are categorised as forest by a Forest Fire Management Victoria fuel layer in a grid with the fire start location at the centre and apothem 3 km.
| `shrub_density_3km` | Shrub density (3 km) | - | Fraction of (30m by 30 m) grid cells that are categorised as shrub by a Forest Fire Management Victoria fuel layer in a grid with the fire start location at the centre and apothem 3 km.
| `noveg_density_3km` | No fuel density (3 km) | - | Fraction of (30m by 30 m) grid cells that are categorised as no fuel by a Forest Fire Management Victoria fuel layer in a grid with the fire start location at the centre and apothem 3 km.
| `primary_fuel_type` | Primary fuel type | - | Category of grass, forest, shrub or no fuel based on which density (3 km) is the largest.
| `distance_to_interface` | Distance to grass-bush interface | - | Manhattan distance, counting the number of 30 m grid cells to the boundary between the grass and bush (forest or shrub). If the fire start location is in grass, it is the distance to bush and vice versa.  If the fire starts in no vegetation, the value is set to zero. Calculated from a Forest Fire Management Victoria vegetation layer.
| `elevation_m` | Elevation | m | Distance above sea level, 30 m grid, Vicmap.
| `ruggedness_average_3km` | Ruggedness average | - | Average ruggedness in a 3km apothem square around the fire start location. Ruggedness calculated according to (Reily Shawn et al. 1999).
| `building_density_3km` | Building denesity (3 km) | - | Number of buildings within a 3 km radius of the fire start location. This variable captures whether the fire started in a town or not. Calculated from Vicmap buildings data.
| `building_density_20km` | Building denesity (20 km) | - | Number of buildings within a 20 km radius of the fire start location. This variable captures the remoteness of the fire, as more buildings in a 20 km radius means there is likely to be a faster response time. Calculated from Vicmap buildings data.
| `road_distance_m` | Distance to road | m | Distance as the crow flies to the nearest road from the fire report location. Calculated from Vicmap transport road line data.
| `FFDI` | Forest fire danger index | - | MacArthur's forest fire danger index from the most recent forecast before the fire report time from the Australian Digital Forecast Data, Bureau of Meteorology.
| `FFDI_historical` | Forest fire danger index (historical) | - | MacArthur's forest fire danger index from the VicClim grid value closest to the reported fire time and location.
| `GFDI` | Grass fire danger index | - | MacArthur's grass fire danger index from the most recent forecast before the fire report time from the Australian Digital Forecast Data, Bureau of Meteorology.
| `GFDI_historical` | Grass fire danger index (historical) | - | MacArthur's grass fire danger index from the VicClim grid value closest to the reported fire time and location.
| `uncontrolled_within_2_hrs` | Uncontrolled within 2 hours | - | Indicator variable taking the value 0 if the fire is controlled within 2 hours of it being reported and 1 otherwise.
| `uncontrolled_within_4_hrs` | Uncontrolled within 4 hours | - | Indicator variable taking the value 0 if the fire is controlled within 4 hours of it being reported and 1 otherwise.
| `uncontrolled_within_5_ha` | Uncontrolled within 5 hectares | - | Indicator variable taking the value 0 if the final fire perimeter is at most 5 hectares and 1 otherwise.
| `uncontrolled_within_100_ha` | Uncontrolled within 100 hectares | - | Indicator variable taking the value 0 if the final fire perimeter is at most 100 hectares and 1 otherwise.
