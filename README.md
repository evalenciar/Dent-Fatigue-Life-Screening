# Dent Fatigue Life Screening

A Python-based toolkit for performing fatigue life assessment of pipeline dents using MD-5-3, MD-2-4, and MD-4-9 methodology, rainflow cycle counting, and pressure history analysis.

## Overview

This project provides a comprehensive solution for evaluating the remaining fatigue life of pipeline dents with interacting features by combining:
- ILI (In-Line Inspection) mechanical caliper data processing
- Pressure history analysis and rainflow cycle counting
- MD-5-3, MD-2-4, and MD-4-9 fatigue assessment methodology
- Multiple S-N curve standards (ABS, BS 7608, DNV)

## Features

- **MD-4-9 Profile Generation**: Process raw ILI caliper data to create standardized dent profiles
- **Rainflow Cycle Counting**: Analyze pressure history data to determine equivalent fatigue cycles
- **Fatigue Life Assessment**: Calculate remaining life using industry-standard S-N curves
- **Multiple Assessment Levels**: Support for Level 0, 0.5, 0.75, 1, and 2 assessments
- **Scale Factor Integration**: Include confidence, CPS, and other safety factors
- **Flexible Data Input**: Support for various data formats and sources

## Installation

### Prerequisites

- Python 3.8+
- Required packages (install via pip):

```bash
pip install pandas numpy matplotlib rainflow
```

### Setup

1. Clone or download this repository
2. Install the required dependencies
3. Import the modules in your Python environment

## Quick Start

### 1. Data Preparation

#### ILI Caliper Data Format
Your ILI caliper data should be in CSV format with:
- **Columns**: Caliper channel orientations in degrees (N channels)
- **Rows**: Axial displacements in inches (M entries, where M > N)
- **Values**: Absolute internal radius in inches

Example format:
```
        0.0    45.0    90.0    135.0   ...
0.0     6.125  6.130   6.128   6.127   ...
0.1     6.124  6.129   6.126   6.125   ...
0.2     6.123  6.128   6.124   6.123   ...
...     ...    ...     ...     ...     ...
```

#### Pressure History Data Format
Your pressure history data should be in CSV format with:
- **Column 1**: Absolute time
- **Column 2**: Upstream pressure (psig)
- **Column 3**: Downstream pressure (psig)

Example format:
```
Time                    US_Pressure  DS_Pressure
2023-01-01 00:00:00    1000.5       995.2
2023-01-01 01:00:00    1002.3       997.1
...                    ...          ...
```

### 2. Basic Usage

```python
import cycle_counting as cc
import md_profiles as mdp
import md_fatigue as mdf
import pandas as pd
import numpy as np

# Load your data
df_dent = pd.read_csv("your_caliper_data.csv", index_col=0, dtype=float)
df_dent.columns = df_dent.columns.astype(float)
df_pressure = pd.read_csv("your_pressure_data.csv", index_col=0)

# Define dent metadata
dent_metadata = {
    "OD": 12.75,      # Outside diameter (inches)
    "WT": 0.250,      # Wall thickness (inches) 
    "SMYS": 52000,    # Specified minimum yield strength (psi)
    "Lx": 1000,       # Dent location length (feet)
    "hx": 30,         # Dent location height (feet)
    "SG": 0.8,        # Product specific gravity
    "L1": 500,        # Upstream segment length (feet)
    "L2": 1500,       # Downstream segment length (feet)
    "h1": 10,         # Upstream segment height (feet)
    "h2": 50,         # Downstream segment height (feet)
    "D1": 12.75,      # Upstream segment diameter (inches)
    "D2": 12.75,      # Downstream segment diameter (inches)
}

# Step 1: Generate MD-4-9 Profiles
dp = mdp.DentProfiles(df_dent, dent_metadata["OD"], dent_metadata["WT"])
print(dp)  # View profile summary
dp.graph("Axial")  # Generate profile visualization

# Step 2: Perform Rainflow Cycle Counting
dc = cc.DentCycles(
    dent_metadata["OD"], dent_metadata["WT"], dent_metadata["SMYS"],
    dent_metadata["Lx"], dent_metadata["hx"], dent_metadata["SG"],
    dent_metadata["L1"], dent_metadata["L2"], dent_metadata["h1"],
    dent_metadata["h2"], dent_metadata["D1"], dent_metadata["D2"]
)

Neq_SSI, Neq_CI, Neq_binned, cycles, cycles_binned = dc.process_liquid(
    [df_pressure.iloc[:, 0].to_numpy(), df_pressure.iloc[:, 1].to_numpy()],
    df_pressure.index.to_numpy(dtype=np.datetime64)
)

# Step 3: Calculate Remaining Fatigue Life
curve_selection = {
    "Category": "BS",  # Options: "ABS", "BS", "DNV"
    "Curve": "D",      # Curve designation
    "SD": 0            # Standard deviation
}

calc_restraint, rp_results, fatigue_results = mdf.get_fatigue_life(
    dc, dp, cycles, curve_selection
)

# Display results
print(f"Calculated Restraint: {calc_restraint}")
for level, sf_results in fatigue_results.items():
    for sf_condition, life_years in sf_results.items():
        print(f"Level {level}, {sf_condition}: {life_years:.3f} years")
```

## Modules

### `cycle_counting.py`
Manages pressure history analysis and cycle counting:
- `DentCycles`: Main class for pressure analysis
- Rainflow cycle counting implementation
- Pressure interpolation at dent location
- Equivalent cycle calculations (SSI, CI)
- Custom binning (default is 28 bins from MD-4-9)

### `md_profiles.py`
Handles ILI caliper data processing and MD-4-9 profile generation:
- `DentProfiles`: Main class for processing dent geometry
- Profile visualization and analysis
- Geometric parameter extraction

### `md_fatigue.py`
Performs fatigue life assessment:
- Multi-level fatigue analysis (Level 0 through Level 2)
- S-N curve integration (ABS, BS, DNV standards)
- Scale factor applications
- Remaining life calculations

## Assessment Levels

- **Level 0**: Basic dent depth assessment
- **Level 0.5**: Enhanced geometric analysis
- **Level 0.75**: Advanced profile-based assessment
- **Level 1**: Stress-based analysis (requires ILI pressure data)
- **Level 2**: Full stress range analysis with cycle counting
- **Level 2 MD-2-4**: Enhanced binned analysis
- **Level 2 MD-2-4 Unbinned**: Detailed unbinned cycle analysis

## Supported S-N Curves

### ABS (American Bureau of Shipping)
- Guide for the Fatigue Assessment of Offshore Structure 2020, Table 1

### BS 7608:2014  
- Guide to fatigue design and assessment of steel products, Table 18

### DNV (Det Norske Veritas)
- DNVGL-RP-C203 Fatigue design of offshore steel structures, Table 2-1

## Output

The analysis provides:
- **Calculated Restraint**: Determined restraint condition
- **RP Values**: Quadrant-specific restraint parameters
- **Fatigue Life Results**: Remaining life estimates for each assessment level
- **Scale Factors**: Applied safety and confidence factors
- **Cycle Counts**: Detailed cycle counting results

## Example Workflow

See `Example_Workflow.ipynb` for a complete step-by-step tutorial demonstrating:
1. Data loading and preparation
2. Profile generation and visualization  
3. Pressure history processing
4. Fatigue life calculation
5. Results interpretation

## Data Requirements

### Minimum Required Data
- ILI mechanical caliper data in standardized format
- Pressure history data (upstream and downstream)
- Pipe specifications (OD, WT, SMYS)
- Dent location information

### Optional Data
- ILI pressure data (enables Level 1 and Level 2 assessments)
- Confidence levels and safety factors
- Custom S-N curve parameters

## Contributing

This project follows standard Python development practices. When contributing:
1. Ensure proper documentation of new features
2. Include example usage in docstrings
3. Add appropriate error handling
4. Update tests as needed

## License

GNU General Public License v3.0

## Support

For questions, issues, or contributions, please contact me at [emmanuel.valencia@softnostics.com](mailto:emmanuel.valencia@softnostics.com)

## References

- API 1176: Assessment and Management of Cracking in Pipelines
- API 1183: Assessment and Management of Pipeline Dents  
- ASME B31.8: Gas Transmission and Distribution Piping Systems
- Various S-N curve standards (ABS, BS 7608, DNV-GL)
