# coding: utf-8
"""
Implements the rainflow cycle counting algorithm for pressure history data 
analysis according to API RP 1183 Section 6.6.3.1 Rainflow Counting.
"""

import numpy as np
import pandas as pd
import rainflow
import matplotlib.pyplot as plt
import math
import os

# Turn off interactive plotting
plt.ioff()

# Default MD-4-9 bin values
pmin_list = [10,10,10,10,10,10,10,20,20,20,20,20,20,30,30,30,30,30,40,40,40,40,50,50,50,60,60,70]
pmax_list = [20,30,40,50,60,70,80,30,40,50,60,70,80,40,50,60,70,80,50,60,70,80,60,70,80,70,80,80]
prange_list = [10,20,30,40,50,60,70,10,20,30,40,50,60,10,20,30,40,50,10,20,30,40,10,20,30,10,20,10]
pmean_list = [15,20,25,30,35,40,45,25,30,35,40,45,50,35,40,45,50,55,45,50,55,60,55,60,65,65,70,75]
# Create a dictionary containing pmin, pmax, prange, pmean lists for easy access. Organize it by bin number (1-28).
default_md49_bins = {i+1: {"pmin": pmin_list[i], "pmax": pmax_list[i], "prange": prange_list[i], "pmean": pmean_list[i]} for i in range(len(pmin_list))}

class DentCycles:
    def __init__(self, 
                 OD:float, 
                 WT:float, 
                 SMYS:float, 
                 Lx:float, 
                 hx:float, 
                 SG:float, 
                 L1:float, 
                 L2:float, 
                 h1:float, 
                 h2:float, 
                 D1:float, 
                 D2:float, 
                 dent_category:str=None, 
                 dent_ID:str=None, 
                 units:str='Imperial',
                 M:float=3,
                 min_range:float=5, 
                 **kwargs
                 ):
        """
        Initialize the PressureHistory object with pipe and dent parameters.
        Parameters
        ----------
        OD : float
            Outside Diameter of the pipe (inches or meters)
        WT : float
            Wall Thickness of the pipe (inches or meters)
        SMYS : float
            Specified Minimum Yield Strength of the pipe (psi or MPa)
        Lx : float
            Location of point analysis (feet or meters)
        hx : float
            Elevation of point analysis (feet or meters)
        SG : float
            Specific Gravity of the product (dimensionless)
        L1 : float
            Location of upstream discharge station (feet or meters)
        L2 : float
            Location of downstream suction station (feet or meters)
        h1 : float
            Elevation of upstream discharge station (feet or meters)
        h2 : float
            Elevation of downstream suction station (feet or meters)
        D1 : float
            Pipe diameter of segment between L1 and Lx (inches or meters)
        D2 : float
            Pipe diameter of segment between Lx and L2 (inches or meters)
        dent_category : str, optional
            Category of the dent (e.g., "Dent", "Dent with Gouge", etc.). Default is None.
        dent_ID : str, optional
            Unique identifier for the dent feature. Default is None.
        units : str, optional
            Units of measurement ('Imperial' or 'Metric'), by default 'Imperial'
        M : float, optional
            S-N curve slope. Default is 3.0.
        min_range : float, optional
            Minimum pressure range to consider for analysis (psi or MPa). Default is 5-psig.
        **kwargs
            Additional keyword arguments.
        """
        self.dent_category = dent_category
        self.dent_ID = dent_ID
        self.OD = OD
        self.WT = WT
        self.SMYS = SMYS
        self.Lx = Lx
        self.hx = hx
        self.SG = SG
        self.L1 = L1
        self.L2 = L2
        self.h1 = h1
        self.h2 = h2
        self.D1 = D1
        self.D2 = D2
        self.units = units
        self.M = M
        self.min_range = min_range

        # Placeholders for future attributes
        self._Neq_SSI = None
        self._Neq_CI = None
        self._Neq_binned = None
        self._cycles_binned = None
        self._range_max = None
        self._range_min = None
        self._range_avg = None
        self._mean_max = None
        self._mean_min = None
        self._mean_avg = None

        # If units are metric, convert to imperial for calculations
        if units.lower() == 'metric':
            self.OD = OD / 0.0254  # Convert meters to inches
            self.WT = WT / 0.0254  # Convert meters to inches
            self.SMYS = SMYS * 0.145038  # Convert MPa to psi
            self.Lx = Lx * 3.28084  # Convert meters to feet
            self.hx = hx * 3.28084  # Convert meters to feet
            self.L1 = L1 * 3.28084  # Convert meters to feet
            self.L2 = L2 * 3.28084  # Convert meters to feet
            self.h1 = h1 * 3.28084  # Convert meters to feet
            self.h2 = h2 * 3.28084  # Convert meters to feet
            self.D1 = D1 / 0.0254  # Convert meters to inches
            self.D2 = D2 / 0.0254  # Convert meters to inches
            self.min_range = min_range * 0.145038  # Convert MPa to psi

    @property
    def statistics(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing statistics of the rainflow cycles.
        Returns
        -------
        pd.DataFrame
            DataFrame containing statistics of the rainflow cycles.
        """
        # If statistics have not been calculated yet, raise an error
        if self._range_max is None:
            raise ValueError("Rainflow cycle statistics have not been calculated.")
        data = {
            'Parameter': ['Maximum Pressure Range', 'Minimum Pressure Range', 'Average Pressure Range', 
                          'Maximum Pressure Mean', 'Minimum Pressure Mean', 'Average Pressure Mean',
                          'Neq using Spectrum Severity Indicator (SSI)', 'Neq using Cyclic Index (CI)', 'Neq using Binned Cycles'],
            'Value': [f"{self._range_max:.2f}", f"{self._range_min:.2f}", f"{self._range_avg:.2f}",
                      f"{self._mean_max:.2f}", f"{self._mean_min:.2f}", f"{self._mean_avg:.2f}",
                      f"{self._Neq_SSI:.2f}", f"{self._Neq_CI:.2f}", f"{self._Neq_binned:.2f}"],
            'Unit': ['psig', 'psig', 'psig', 'psig', 'psig', 'psig', 'cycles/year (13ksi ref. stress)', 'cycles/year (37.58ksi ref. stress)', 'cycles/year (13ksi ref. stress)']
        }
        df = pd.DataFrame(data)
        return_string = (
            f"Rainflow Cycle Statistics:\n"
            f"{df.to_string(index=False)}"
        )
        return print(return_string)

    @property
    def cycles_binned(self) -> dict:
        """
        Returns a dictionary containing the binned rainflow cycles.
        Returns
        -------
        dict
            Dictionary containing the binned rainflow cycles.
        """
        # If binned cycles have not been calculated yet, raise an error
        if self._cycles_binned is None:
            raise ValueError("Binned rainflow cycles have not been calculated.")
        return self._cycles_binned
    
    def __repr__(self):
        data = {
            'Parameter': ['Outside Diameter (OD)', 'Wall Thickness (WT)', 'Specified Minimum Yield Strength (SMYS)', 
                          'Service Years', 'S-N Curve Slope (M)', 'Minimum Pressure Range', 'Location of Point Analysis (Lx)', 
                          'Elevation of Point Analysis (hx)', 'Specific Gravity (SG)', 'Location of Upstream Discharge Station (L1)', 
                          'Location of Downstream Suction Station (L2)', 'Elevation of Upstream Discharge Station (h1)', 
                          'Elevation of Downstream Suction Station (h2)', 'Pipe Diameter at L1 (D1)', 
                          'Pipe Diameter at L2 (D2)', 'Units'],
            'Value': [f"{self.OD:.2f}", f"{self.WT:.2f}", f"{self.SMYS:.2f}", 
                      f"{self.service_years:.2f}", f"{self.M:.2f}", f"{self.min_range:.2f}", f"{self.Lx:.2f}", 
                      f"{self.hx:.2f}", f"{self.SG:.2f}", f"{self.L1:.2f}", f"{self.L2:.2f}", 
                      f"{self.h1:.2f}", f"{self.h2:.2f}", f"{self.D1:.2f}", f"{self.D2:.2f}", self.units],
            'Unit': ['in', 'in', 'psi', 'years', '-', 'psig', 'ft', 'ft', '-', 'ft', 'ft', 'ft', 'ft', 'in', 'in', '-']
        }
        # If a dent ID and category are provided, add them to the data at the top
        if self.dent_ID is not None and self.dent_category is not None:
            data['Parameter'].insert(0, 'Dent ID')
            data['Value'].insert(0, self.dent_ID)
            data['Parameter'].insert(1, 'Dent Category')
            data['Value'].insert(1, self.dent_category)
        # If processing has been done, add Neq values
        if self._Neq_SSI is not None:
            data['Parameter'].extend(['Neq using Spectrum Severity Indicator (SSI)', 'Neq using Cyclic Index (CI)', 'Neq using Binned Cycles'])
            data['Value'].extend([f"{self._Neq_SSI:.2f}", f"{self._Neq_CI:.2f}", f"{self._Neq_binned:.2f}"])
            data['Unit'].extend(['cycles/year (13ksi ref. stress)', 'cycles/year (37.58ksi ref. stress)', 'cycles/year (13ksi ref. stress)'])

        df = pd.DataFrame(data)
        return_string = (
            f"DentCycles Object Summary:\n"
            f"{df.to_string(index=False)}"
        )
        return return_string

    def _statistics(self, cycles: np.ndarray) -> pd.DataFrame:
        """
        Generate statistics from the rainflow cycles.
        Parameters
        ----------
        cycles : np.ndarray
            Array of rainflow cycles with columns: [Pressure Range (psig), Pressure Mean (psig), Cycle Count, Index Start, Index End]
        Returns
        -------
        pd.DataFrame
            DataFrame containing statistics of the rainflow cycles.
        """
        self._range_max = np.max(cycles[:, 0])
        self._range_min = np.min(cycles[:, 0])
        self._range_avg = np.mean(cycles[:, 0])
        self._mean_max = np.max(cycles[:, 1])
        self._mean_min = np.min(cycles[:, 1])
        self._mean_avg = np.mean(cycles[:, 1])

        return

    def process_liquid(self, P_list: list[np.ndarray], P_time: np.ndarray, results_path:str=os.getcwd(), binning_dict: dict=None, create_figure:bool=False, create_histogram:bool=False, save_history:bool=False, save_cycles:bool=False, save_cycles_binned:bool=False) -> tuple[float, float, float, np.ndarray, dict]:
        """
        Perform rainflow analysis on liquid pressure history data.
        Parameters
        ----------
        P_list : list of arrays
            List containing two arrays: [upstream pressure array, downstream pressure array], both in psig
        P_time : np.ndarray
            Array of datetime objects corresponding to the pressure data points
        results_path : str, optional
            Path to save results, by default os.getcwd()
        binning_dict : dict, optional
            Dictionary containing custom pressure bin values (pmin, pmax, prange, pmean) in %SMYS, by default None. If none is provided, the 28 bins from the MD-4-9 will be used.
        create_figure : bool, optional
            Whether to create and save a pressure history figure, by default False
        create_histogram : bool, optional
            Whether to create and save a rainflow cycle histogram figure, by default False
        save_history : bool, optional
            Whether to save the interpolated pressure history data as a CSV file, by default False
        save_cycles : bool, optional
            Whether to save the rainflow cycles as a CSV file, by default False
        save_cycles_binned : bool, optional
            Whether to save the binned rainflow cycles as a CSV file, by default False
        Returns
        -------
        Neq_SSI : float
            Number of equivalent cycles per year using continuous data and the Spectrum Severity Indicator (SSI) having a reference stress of 13 ksi
        Neq_CI : float
            Number of equivalent cycles per year using continuous data and the Cyclic Index (CI) having a reference stress of 37.58 ksi
        Neq_binned : float
            Number of equivalent cycles per year using binned data (by default, the 28 bins from the MD-4-9 unless custom bins are provided) having a reference stress of 13 ksi
        cycles : np.ndarray
            Array of rainflow cycles with columns: [Pressure Range (psig), Pressure Mean (psig), Cycle Count, Index Start, Index End]
        cycles_binned : np.ndarray
            Array of rainflow cycle counts grouped into discrete bins (by default, the 28 bins from the MD-4-9 unless custom bins are provided)
        """
        # Error handling for inputs
        if len(P_list) != 2:
            raise ValueError("P_list must contain exactly two arrays: [upstream pressure array, downstream pressure array].")
        if not all(isinstance(arr, np.ndarray) for arr in P_list):
            raise TypeError("Both elements in P_list must be numpy arrays.")
        if P_list[0].shape != P_list[1].shape:
            raise ValueError("Upstream and downstream pressure arrays must have the same shape.")
        # Check that P_time length matches pressure arrays length
        if P_time.shape[0] != P_list[0].shape[0]:
            raise ValueError("P_time array length must match the length of the pressure arrays.")
        # Check that P_time is a datetime array
        if not np.issubdtype(P_time.dtype, np.datetime64):
            raise TypeError("P_time array must be of datetime type.")
        
        # Determine the service years from the pressure time data
        self.service_years = (P_time[-1] - P_time[0]).astype('timedelta64[D]').item().days / 365.25
        
        # Determine the operational pressures at the dent location. Taken from Equation (5) in API 1183 Section 6.6.3.1 Rainflow Counting
        P = Px(P_list[0], P_list[1], self.Lx, self.hx, self.SG, self.L1, self.L2, self.h1, self.h2, self.D1, self.D2)

        # Rainflow Analysis using Python package 'rainflow'. Output is in format: Pressure Range [psig], Pressure Mean [psig], Cycle Count, Index Start, Index End
        cycles = pd.DataFrame(rainflow.extract_cycles(P)).to_numpy()

        # Filter the cycles array based on the min_range using the Pressure Range (first column)
        cycles = cycles[cycles[:, 0] > self.min_range]

        # Calculate the SSI, CI, and SSI MD49
        Neq_SSI = equivalent_cycles(cycles, self.OD, self.WT, self.service_years, self.M, self.min_range, "ssi")
        Neq_CI = equivalent_cycles(cycles, self.OD, self.WT, self.service_years, self.M, self.min_range, "ci")
        if binning_dict:
            Neq_binned, cycles_binned = bin_custom(cycles, self.OD, self.WT, self.SMYS, self.service_years, self.M, binning_dict, self.min_range)
        else:
            Neq_binned, cycles_binned = bin_MD49(cycles, self.OD, self.WT, self.SMYS, self.service_years, self.M, self.min_range)
        # Save to self attributes. To preserve memory, only cycles statistics are saved (not the full cycles array)
        self._Neq_SSI = Neq_SSI
        self._Neq_CI = Neq_CI
        self._Neq_binned = Neq_binned
        # Update the default_md49_bins dictionary with the cycle counts
        self._cycles_binned = default_md49_bins.copy()
        for bin_num, bin_data in self._cycles_binned.items():
            bin_data["cycle_count"] = cycles_binned[bin_num] if bin_num < len(cycles_binned) else 0
        # Calculate and save statistics
        self._statistics(cycles)

        # Save outputs if specified
        if save_history:
            df_P = pd.DataFrame(data=P, columns=['Pressure (psig)'], index=P_time)
            df_P.to_csv(os.path.join(results_path, f"Feature {self.dent_ID} Interpolated_Pressure_History_Data.csv" if self.dent_ID is not None else "Interpolated_Pressure_History_Data.csv"), header=False, index=False)
        if save_cycles:
            np.savetxt(os.path.join(results_path, f"Feature {self.dent_ID} Cycles.csv" if self.dent_ID is not None else "Cycles.csv"), cycles, delimiter=',')
        if save_cycles_binned:
            np.savetxt(os.path.join(results_path, f"Feature {self.dent_ID} Cycles_Binned.csv" if self.dent_ID is not None else "Cycles_Binned.csv"), cycles_binned, delimiter=',')
        if create_figure:
            self.graph_liquid(P, P_time, results_path)
        if create_histogram:
            self.graph_histogram(cycles, results_path)

        return Neq_SSI, Neq_CI, Neq_binned, cycles, cycles_binned

    def graph_liquid(self, P: np.ndarray, P_time: np.ndarray, results_path:str=None):
        # Save the interpolated pressure history 
        fig, sp = plt.subplots(figsize=(8,4), dpi=240)
        suptitle_str = f'Pressure History for {self.dent_category} Feature {self.dent_ID}' if self.dent_ID is not None and self.dent_category is not None else 'Pressure History'
        fig.suptitle(suptitle_str, fontsize=16)
        sp.scatter(P_time, P, s=0.1)
        sp.grid(color='lightgray', alpha=0.5, zorder=1)
        sp.set_ylim((0, max(1750.00, math.ceil(np.nanmax(P) / 250) * 250))) # Set y-axis limit to a maximum of 2000 or the next multiple of 250 above the max pressure
        sp.set_ylabel('Interpolated Pressure (psig)')
        sp.set_xlabel('Date Time')
        if results_path:
            filename_str = f"Feature_{self.dent_ID}_Interpolated_Pressure_History" if self.dent_ID is not None else "Interpolated_Pressure_History"
            fig.savefig(os.path.join(results_path, filename_str + ".png"))
            plt.close(fig)

    def graph_histogram(self, cycles: np.ndarray, results_path:str=None):
        # Save the rainflow histogram 
        fig, sp = plt.subplots(figsize=(8,4), dpi=240)
        suptitle_str = f'Rainflow Cycle Histogram for {self.dent_category} Feature {self.dent_ID}' if self.dent_ID is not None and self.dent_category is not None else 'Rainflow Cycle Histogram'
        fig.suptitle(suptitle_str, fontsize=16)
        sp.hist(cycles[:,0], bins=50, edgecolor='black')
        sp.grid(color='lightgray', alpha=0.5, zorder=1)
        sp.set_xlabel('Pressure Range (psig)')
        sp.set_ylabel('Cycle Count')
        if results_path:
            filename_str = f"Feature_{self.dent_ID}_Rainflow_Cycle_Histogram" if self.dent_ID is not None else "Rainflow_Cycle_Histogram"
            fig.savefig(os.path.join(results_path, filename_str + ".png"))
            plt.close(fig)

def Px(P1:np.ndarray, P2:np.ndarray, Lx:float, hx:float, SG:float, L1:float, L2:float, h1:float, h2:float, D1:float, D2:float) -> np.ndarray:
    """
    Taken from API 1176
    
    Note: the version in API 1183 Section 6.6.3.1 Rainflow Counting Equation (5) is incorrect.
    
    Parameters
    ----------
    P1 : np.ndarray
        the upstream discharge pressure, psig
    P2 : np.ndarray
        the downstream suction pressure, psig
    Lx : float
        the location of point analysis, ft
    hx : float
        the elevation of point analysis, ft
    K : float
        SG x (0.433 psi/ft), where SG = specific gravity of product
    L1 : float
        the location of upstream discharge station, ft
    L2 : float
        the location of downstream suction station, ft
    h1 : float
        the elevation of upstream discharge station, ft
    h2 : float
        the elevation of downstream suction station, ft
    D1 : float
        the pipe diameter of segment between L1 and Lx, in
    D2 : float
        the pipe diameter of segment between Lx and L2, in

    Returns
    -------
    The intermediate pressure point between pressure sources, psig

    """
    # Error handling for inputs
    if not (P1.shape == P2.shape):
        raise ValueError("P1 and P2 must have the same shape.")
    K = SG * 0.433  # Convert SG to psi/ft
    Px = (P1 + K*h1 - P2 - K*h2)*(1/(((Lx - L1)*D2**5)/((L2 - Lx)*D1**5) + 1)) - K*(hx - h2) + P2
    
    return Px

def bin_custom(cycles, OD: float, WT: float, SMYS: float, service_years: float, M: float, binning_dict: dict, min_range: float = 5) -> tuple[float, dict]:
    '''
    Use the custom pressure bins defined in binning_dict to sum the cycles into the bins.
    Parameters
    ----------
    cycles : array of floats
        the array output from the rainflow analysis, with columns: [Pressure Range (psig), Pressure Mean (psig), Cycle Count, Index Start, Index End]
    OD : float
        the outside diameter of the pipe, in
    WT : float
        the wall thickness of the pipe, in
    SMYS : float
        the specified minimum yield strength of the pipe, psi
    service_years : float
        the number of years the pressure history represents, years
    M : float
        the slope of the S-N curve, typically 3.0 for steel
    binning_dict : dict
        dictionary containing pressure bin values (pmin, pmax, prange, pmean) in %SMYS
    min_range : float, optional
        the minimum pressure range to consider for the analysis, default is 5 psi

    Returns
    -------
    (bin_SSI : float, bin_cycles : dict)
        A tuple containing the total damage equivalent cycles and a dictionary of the custom bins with their respective cycle counts.
    '''
    # Error handling for inputs
    if not isinstance(binning_dict, dict):
        raise TypeError("binning_dict must be a dictionary.")
    if not (cycles.shape[1] >= 3):
        raise ValueError("cycles array must have at least 3 columns: [Pressure Range, Pressure Mean, Cycle Count]. Use the output from the rainflow module, for example: cycles = pandas.DataFrame(rainflow.extract_cycles(P)).to_numpy()")
    
    # Filter out cycles below the minimum range
    custom_cycles = cycles.copy()
    for i in range(len(custom_cycles)):
        if custom_cycles[i,0] < min_range: 
            custom_cycles[i,2] = 0
    # Convert pressure values into units of % SMYS
    custom_cycles[:,0] = 100*custom_cycles[:,0]*OD/(2*WT)/SMYS
    custom_cycles[:,1] = 100*custom_cycles[:,1]*OD/(2*WT)/SMYS
    
    # Make the second level keys lowercase for consistency, and the first level keys integers
    binning_dict = {int(k): {kk.lower(): vv for kk, vv in v.items()} for k, v in binning_dict.items()}
    # Create an empty dictionary to hold the cycle counts for each bin
    bin_cycles = {k: 0 for k in binning_dict.keys()}
    
    # Create groups of equivalent Prange bins which will then be used for the iterative processing
    bin_groups = {}
    for bin_num, bin_vals in binning_dict.items():
        if bin_vals['prange'] not in bin_groups:
            bin_groups[bin_vals['prange']] = []
        bin_groups[bin_vals['prange']].append({int(bin_num): bin_vals["pmean"]})
    # Sort the bin_groups by the prange key
    bin_groups = dict(sorted(bin_groups.items()))

    # Iterate through every pressure range cycle, and find the appropriate bin to add the cycle count to using the bin_groups
    for i, press_range in enumerate(custom_cycles[:, 0]):
        for bin_range, bin_vals in bin_groups.items():
            if press_range <= bin_range:
                for i, bin_val in enumerate(bin_vals):
                    if i != len(bin_vals) - 1 and (custom_cycles[i,1] <= (list(bin_vals[i].values())[0] + list(bin_vals[i + 1].values())[0])/2):  # If the mean is less than the midpoint between this bin and the next bin, add it here
                        bin_cycles[list(bin_val.keys())[0]] += custom_cycles[i,2]
                        break
                    if i == len(bin_vals) - 1:  # Last bin, so just add it here
                        bin_cycles[list(bin_val.keys())[0]] += custom_cycles[i,2]
                        break
                break

    # Calculate the damage equivalent cycles for the custom bins
    SSI_ref_stress = 13000  # psi
    # The Neq SSI = (Prange_%SMYS * SMYS / SSI_ref_stress) ^ M * Cycles
    Prange_pct_smys = np.array([v['prange'] for v in binning_dict.values()])
    Neq_SSI = sum(((Prange_pct_smys* SMYS / SSI_ref_stress) ** M) * np.array(list(bin_cycles.values()))) / service_years
    return Neq_SSI, bin_cycles

def bin_MD49(cycles:np.ndarray, OD:float, WT:float, SMYS:float, service_years:float, M:float, min_range:float=5) -> tuple[float, np.ndarray]:
    """
    Bin the rainflow cycles into the MD-4-9 bins as defined in API 1183.
    Parameters
    ----------
    cycles : np.ndarray
        the array output from the rainflow analysis, with columns: [Pressure Range (psig), Pressure Mean (psig), Cycle Count, Index Start, Index End]
    OD : float
        the outside diameter of the pipe, in
    WT : float
        the wall thickness of the pipe, in
    SMYS : float
        the specified minimum yield strength of the pipe, psi
    service_years : float
        the number of years the pressure history represents, years
    M : float
        the slope of the S-N curve, typically 3.0 for steel
    min_range : float, optional
        the minimum pressure range to consider for the analysis, default is 5 psi
    Returns
    -------
    MD49_SSI : float
        MD49 Spectrum Severity Indicator equivalent cycles per year
    MD49_bins : np.ndarray
        Array of MD49 bins with cycle counts
    """
    # Error handling for inputs
    if not (cycles.shape[1] >= 3):
        raise ValueError("cycles array must have at least 3 columns: [Pressure Range, Pressure Mean, Cycle Count]. Use the output from the rainflow module, for example: cycles = pandas.DataFrame(rainflow.extract_cycles(P)).to_numpy()")
    
    # Create an empty array for all the MD-4-9 bins
    MD49_bins = np.zeros(28)
    MD49_P_range = np.array([10,20,30,40,50,60,70,
                             10,20,30,40,50,60,
                             10,20,30,40,50,
                             10,20,30,40,
                             10,20,30,
                             10,20,
                             10])
    
    # Remove any MD49_cycles that have a pressure range below the minimum range
    MD49_cycles = cycles.copy()
    for i, val in enumerate(MD49_cycles[:,0]):
        if MD49_cycles[i,0] < min_range: 
            MD49_cycles[i,2] = 0
    
    # Reference Stress Ranges
    SSI_ref_stress = 13000  # psi
    
    # Convert pressure range into units of % SMYS
    MD49_cycles[:,0] = 100*MD49_cycles[:,0]*OD/(2*WT)/SMYS
    MD49_cycles[:,1] = 100*MD49_cycles[:,1]*OD/(2*WT)/SMYS
    
    # Iterate through every pressure range cycle
    for i, _ in enumerate(MD49_cycles[:, 0]):
        # Pressure range: 0 - 10% SMYS
        if MD49_cycles[i, 0] <= 10.0: #if range is 0 - 10% SMYS
            if MD49_cycles[i, 1] <= 20.0: #if mean is 0 - 20% SMYS
                MD49_bins[0] = MD49_bins[0] + MD49_cycles[i, 2] #BIN #1
            elif MD49_cycles[i, 1] <= 30.0: #if mean is 20 - 30% SMYS
                MD49_bins[7] = MD49_bins[7] + MD49_cycles[i, 2] #BIN #8
            elif MD49_cycles[i, 1] <= 40.0: #if mean is 30 - 40% SMYS
                MD49_bins[13] = MD49_bins[13] + MD49_cycles[i, 2] #BIN #14
            elif MD49_cycles[i, 1] <= 50.0: #if mean is 40 - 50% SMYS
                MD49_bins[18] = MD49_bins[18] + MD49_cycles[i, 2] #BIN #19
            elif MD49_cycles[i, 1] <= 60.0: #if mean is 50 - 60% SMYS
                MD49_bins[22] = MD49_bins[22] + MD49_cycles[i, 2] #BIN #23
            elif MD49_cycles[i, 1] <= 70.0: #if mean is 60 - 70% SMYS
                MD49_bins[25] = MD49_bins[25] + MD49_cycles[i, 2] #BIN #26
            else: #if mean is >70% SMYS
                MD49_bins[27] = MD49_bins[27] + MD49_cycles[i, 2] #BIN #28
        # Pressure range: 10 - 20% SMYS
        elif MD49_cycles[i, 0] <= 20.0: #if range is 10 - 20% SMYS
            if MD49_cycles[i, 1] <= 25.0: #if mean is 0 - 25% SMYS
                MD49_bins[1] = MD49_bins[1] + MD49_cycles[i, 2] #BIN #2
            elif MD49_cycles[i, 1] <= 35.0: #if mean is 25 - 35% SMYS
                MD49_bins[8] = MD49_bins[8] + MD49_cycles[i, 2] #BIN #9
            elif MD49_cycles[i, 1] <= 45.0: #if mean is 35 - 45% SMYS
                MD49_bins[14] = MD49_bins[14] + MD49_cycles[i, 2] #BIN #15
            elif MD49_cycles[i, 1] <= 55.0: #if mean is 45 - 55% SMYS
                MD49_bins[19] = MD49_bins[19] + MD49_cycles[i, 2] #BIN #20
            elif MD49_cycles[i, 1] <= 65.0: #if mean is 55 - 65% SMYS
                MD49_bins[23] = MD49_bins[23] + MD49_cycles[i, 2] #BIN #24
            else: #if mean is >65% SMYS
                MD49_bins[26] = MD49_bins[26] + MD49_cycles[i, 2] #BIN #27
        # Pressure range: 20 - 30% SMYS
        elif MD49_cycles[i, 0] <= 30.0: #if range is 20 - 30% SMYS
            if MD49_cycles[i, 1] <= 30.0: #if mean is 0 - 30% SMYS
                MD49_bins[2] = MD49_bins[2] + MD49_cycles[i, 2] #BIN #3
            elif MD49_cycles[i, 1] <= 40.0: #if mean is 30 - 40% SMYS
                MD49_bins[9] = MD49_bins[9] + MD49_cycles[i, 2] #BIN #10
            elif MD49_cycles[i, 1] <= 50.0: #if mean is 40 - 50% SMYS
                MD49_bins[15] = MD49_bins[15] + MD49_cycles[i, 2] #BIN #16
            elif MD49_cycles[i, 1] <= 60.0: #if mean is 50 - 60% SMYS
                MD49_bins[20] = MD49_bins[20] + MD49_cycles[i, 2] #BIN #21
            else: #if mean is >60% SMYS
                MD49_bins[24] = MD49_bins[24] + MD49_cycles[i, 2] #BIN #25
        # Pressure range: 30 - 40% SMYS
        elif MD49_cycles[i, 0] <= 40.0: #if range is 30 - 40% SMYS
            if MD49_cycles[i, 1] <= 35.0: #if mean is 0 - 35% SMYS
                MD49_bins[3] = MD49_bins[3] + MD49_cycles[i, 2] #BIN #4
            elif MD49_cycles[i, 1] <= 45.0: #if mean is 35 - 45% SMYS
                MD49_bins[10] = MD49_bins[10] + MD49_cycles[i, 2] #BIN #11
            elif MD49_cycles[i, 1] <= 55.0: #if mean is 45 - 55% SMYS
                MD49_bins[16] = MD49_bins[16] + MD49_cycles[i, 2] #BIN #17
            else: #if mean is >55% SMYS
                MD49_bins[21] = MD49_bins[21] + MD49_cycles[i, 2] #BIN #22
        # Pressure range: 40 - 50% SMYS
        elif MD49_cycles[i, 0] <= 50.0: #if range is 40 - 50% SMYS
            if MD49_cycles[i, 1] <= 40.0: #if mean is 0 - 40% SMYS
                MD49_bins[4] = MD49_bins[4] + MD49_cycles[i, 2] #BIN #5
            elif MD49_cycles[i, 1] <= 50.0: #if mean is 40 - 50% SMYS
                MD49_bins[11] = MD49_bins[11] + MD49_cycles[i, 2] #BIN #12
            else: #if mean is >50% SMYS
                MD49_bins[17] = MD49_bins[17] + MD49_cycles[i, 2] #BIN #18
        # Pressure range: 50 - 60% SMYS
        elif MD49_cycles[i, 0] <= 60.0: #if range is 50 - 60% SMYS
            if MD49_cycles[i, 1] <= 45.0: #if mean is 0 - 45% SMYS
                MD49_bins[5] = MD49_bins[5] + MD49_cycles[i, 2] #BIN #6
            else: #if mean is >45% SMYS
                MD49_bins[12] = MD49_bins[12] + MD49_cycles[i, 2] #BIN #13
        # Pressure range > 60% SMYS
        else: #if pressure range > 60% SMYS
            MD49_bins[6] = MD49_bins[6] + MD49_cycles[i, 2] #BIN #7
            
    # Calculate the MD49 Equivalent Cycles
    MD49_final_cycles = sum(((((MD49_P_range/100)*SMYS)/SSI_ref_stress)**M)*MD49_bins)/service_years
    
    return MD49_final_cycles, MD49_bins

def equivalent_cycles(cycles:np.ndarray, OD:float, WT:float, service_years:float, M:float=3, min_range:float=5, index:str='ssi') -> float:
    """
    Parameters
    ----------
    cycles : np.ndarray
        the array output from the rainflow analysis
    OD : float
        the outside diameter of the pipe, in
    WT : float
        the wall thickness of the pipe, in
    service_years : float
        the period of time for the pressure history data, years
    M : float
        the slope of the S-N curve, typically 3.0 for steel
    min_range : float
        the threshold value for pressure ranges to consider, psig. Anything below this value is filtered out and treated as noise. Default is 5 psig.
    index : str
        SSI or CI. SSI is the Spectrum Severity Indicator (13ksi reference stress), CI is the Cyclic Index (37.58ksi reference stress)

    Returns
    -------
    The equivalent number of cycles per year using the reference stress for the specified index. 

    """
    # Error handling for inputs
    if not (cycles.shape[1] >= 3):
        raise ValueError("cycles array must have at least 3 columns: [Pressure Range, Pressure Mean, Cycle Count]. Use the output from the rainflow module, for example: cycles = pandas.DataFrame(rainflow.extract_cycles(P)).to_numpy()")
    
    equiv_cycles = np.zeros(cycles.shape[0])
    
    # Reference Stress Ranges
    SSI_ref_stress = 13000  # psi
    SSI_ref_press = SSI_ref_stress*2*WT/OD
    CI_ref_stress = 37580   # psi
    CI_ref_press = CI_ref_stress*2*WT/OD
    
    if index.lower() == 'ssi':
        ref_press = SSI_ref_press
    elif index.lower() == 'ci':
        ref_press = CI_ref_press
        
    for i, _ in enumerate(cycles[:,0]):
        if cycles[i,0] > min_range: 
            equiv_cycles[i] = ((cycles[i,0]/ref_press)**M)*cycles[i,2]
        else:
            equiv_cycles[i] = 0
            
    num_cycles = sum(equiv_cycles)/service_years
    
    return num_cycles