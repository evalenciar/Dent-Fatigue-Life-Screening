"""
Determine and evaluate the dent profile for all quadrants according to 
API RP 1183 Section 6.2 Dent Geometry Profile Characterization
"""

import numpy as np
import pandas as pd
import math
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def find_deflection(data: pd.Series, nominal_radius: float, window_size: int = 11, slope_tolerance: float = 0.003, closeness_percentage: float = 5, buffer_offset: int = 5, min_consecutive_deviations: int = 3, circumferential_mode: bool = False, outbound_data: bool = False) -> tuple[int | None, float | None]:
    """
    Finds the index where the caliper begins deflecting in the inbound half of the data. If using outbound data, the function will reverse the data to treat it as inbound.
    
    The data should be the half starting from pristine pipe (with noise and vibrations) moving towards the deepest part of the dent (minimum radius).
    
    Uses a slope approach to detect either sustained deviation from pristine conditions (default) or a local maximum near nominal radius (circumferential_mode).
    
    Parameters
    ----------
    data : pd.Series
        Series of caliper radius measurements for the inbound half, from pristine to dent minimum.
        - Guidance: Split full data at the minimum radius index (e.g., `data = full_data[:min_idx + 1]`). Length must be >= `window_size`. Smooth outliers if extreme.
        - Example: `pd.Series([10.0, 9.99, 9.98, ..., 8.5])`.
    nominal_radius : float
        Expected radius of the pristine pipe.
        - Guidance: Use design specs or mean of pristine section (e.g., first 10–20 points). Adjust for sensor bias if needed.
    window_size : int, optional (default=11)
        Window for Savitzky-Golay filter to smooth derivatives.
        - Guidance: Odd integer (5–21). Use 5–7 for sharp changes, 15–21 for noisy data. Match to data resolution.
    slope_tolerance : float, optional (default=0.001)
        Slope threshold for pristine sections or local maxima.
        - Guidance: 0.001–0.01 for low noise, 0.01–0.1 for high noise. In circumferential_mode, defines zero-slope for maxima.
    closeness_percentage : float, optional (default=5)
        Percentage tolerance for radius closeness to nominal.
        - Guidance: 2–5% for stable data, 5–10% for noisy or variable pipes. Verify with radius histograms.
    buffer_offset : int, optional (default=5)
        Points to subtract from start index for preceding data.
        - Guidance: 3–10 points; smaller for high-resolution, larger for sparse/noisy data.
    min_consecutive_deviations : int, optional (default=3)
        Minimum consecutive deviations (default mode) or consecutive points at local maximum (circumferential_mode).
        - Guidance: 2–5; 2 for sharp deflections/maxima, 3–5 for noisy data.
    circumferential_mode : bool, optional (default=False)
        If True, detects deflection at the first local maximum (slope ≈ 0) near nominal radius. If False, uses sustained deviation.
        - Guidance: Enable for circumferential features causing peaks near nominal radius. Inspect data for such maxima.
    outbound_data : bool, optional (default=False)
        If False, data is from pristine to dent minimum. If True, data is from dent minimum to pristine and will be reversed.
        - Guidance: Ensure data direction matches this flag. Reverse if necessary.
    
    Returns
    -------
    init_idx : int or None
        The relative index in data where deflection starts (with buffer), or None if not detected.
    init_axial : float or None
        The axial location at init_idx, or None if not detected.
    init_radius : float or None
        The radius at init_idx, or None if not detected.
    """
    if not outbound_data:
        if circumferential_mode:
            # Reverse the data to treat it as starting from pristine to dent
            data = data[::-1]
        init_idx, init_radius = _find_deflection_initiation(data.to_numpy(), nominal_radius, window_size, slope_tolerance, closeness_percentage, buffer_offset, min_consecutive_deviations, circumferential_mode)
        init_axial = data.index[init_idx] if init_idx is not None else None

        if circumferential_mode:
            # Convert back to original outbound index (this is the end index)
            init_idx = len(data) - 1 - init_idx

            # Cap to data length
            init_idx = min(len(data) - 1, init_idx)

        return init_idx, init_axial, init_radius
    else:
        # Reverse the data to treat it as starting from pristine to dent
        if not circumferential_mode:
            data = data[::-1]
        init_idx, init_radius = _find_deflection_initiation(data.to_numpy(), nominal_radius, window_size, slope_tolerance, closeness_percentage, buffer_offset, min_consecutive_deviations, circumferential_mode)
        init_axial = data.index[init_idx] if init_idx is not None else None

        if not circumferential_mode:
            # Convert back to original outbound index (this is the end index)
            init_idx = len(data) - 1 - init_idx

            # Cap to data length
            init_idx = min(len(data) - 1, init_idx)

        return init_idx, init_axial, init_radius

def _find_deflection_initiation(data: np.ndarray, nominal_radius: float, window_size: int = 11, slope_tolerance: float = 0.003, closeness_percentage: float = 5, buffer_offset: int = 5, min_consecutive_deviations: int = 3, circumferential_mode: bool = False) -> tuple[int | None, float | None]:
    """
    Internal helper to find the initiation index of deflection assuming the data starts from pristine pipe and moves towards the dent.
    
    In default mode, scans from the left to find the first point where the condition deviates from pristine (flat slope and radius close to nominal) 
    in a sustained manner (for at least min_consecutive_deviations points) to handle noise and vibrations in the pristine section.
    
    In circumferential_mode, identifies the first local maximum (slope ≈ 0) where the radius is close to nominal_radius, suitable for cases where a peak near nominal marks the deflection point (e.g., due to interacting features).
    
    Parameters
    ----------
    data : np.ndarray
        Array of caliper radius measurements along the pipeline section. Must be a 1D array of floating-point values representing radii at sequential points.
        - Guidance: Ensure the data is preprocessed to represent either the inbound (pristine to dent) or reversed outbound (dent to pristine) half. Array length should be at least `window_size`. Smooth extreme outliers manually if necessary, as the function handles moderate noise via the Savitzky-Golay filter.
        - Example: `np.array([10.0, 9.99, 9.98, ..., 8.5])` where 10.0 is nominal radius.
    nominal_radius : float
        Expected radius of the pristine (undamaged) pipe, used as the baseline for detecting deviations in radius and slope.
        - Guidance: Use the pipeline's nominal inner radius from design specs or compute as the mean of a known pristine section (e.g., first/last 10–20 points if flat). Adjust empirically if data shows a sensor bias by inspecting the data plot.
    window_size : int, optional (default=11)
        Window length for the Savitzky-Golay filter to compute a smoothed first derivative (slope), reducing noise effects.
        - Guidance: Must be an odd integer > 1 (e.g., 5, 7, 11, 15). Smaller windows (5–7) preserve local details but are noise-sensitive; larger windows (15–21) smooth more, suitable for noisy data but may delay detection of sharp changes. Choose based on data resolution: larger for high sampling rates, smaller for sparse data. Test by plotting the smoothed derivative.
    slope_tolerance : float, optional (default=0.001)
        Threshold for considering the slope (first derivative) as "flat" (close to zero), indicating a pristine section or a local maximum in circumferential_mode.
        - Guidance: Units are radius change per data point (e.g., mm/sample). Set based on expected noise: 0.001–0.01 for low noise, 0.01–0.1 for high noise. In circumferential_mode, this defines the zero-slope threshold for local maxima. Inspect the derivative of pristine sections to estimate typical slope variations.
    closeness_percentage : float, optional (default=5)
        Percentage tolerance for how close the radius must be to `nominal_radius` to be considered pristine or a valid local maximum.
        - Guidance: Set based on expected radius variation in pristine sections due to noise or manufacturing tolerances. 2–5% is typical for well-calibrated sensors; increase to 10% for noisy data or variable pipe conditions. Check radius histograms of pristine sections to confirm.
    buffer_offset : int, optional (default=5)
        Number of points to subtract from the detected initiation index to include preceding pristine data, ensuring no relevant data is excluded.
        - Guidance: Use 3–10 points depending on data resolution and desired margin. Smaller offsets (3–5) for high-resolution data; larger (5–10) for sparse or noisy data to capture context before deflection. Ensure it doesn’t push the index below 0.
    min_consecutive_deviations : int, optional (default=3)
        In default mode, minimum number of consecutive points that must deviate from pristine conditions to confirm the start of deflection. In circumferential_mode, minimum consecutive points with near-zero slope and radius close to nominal to confirm a local maximum.
        - Guidance: Use 2–5 points; 2 for sharp deflections or clear maxima, 3–5 for noisy data to avoid false positives from brief spikes. Adjust based on inspection of noise patterns or slope behavior near maxima.
    circumferential_mode : bool, optional (default=False)
        If True, detects the deflection point as the global maximum (highest radius) among points with slope ≈ 0 (within slope_tolerance) and radius close to nominal_radius (within closeness_percentage). If False, uses the original sustained-deviation approach.
        - Guidance: Enable for cases where a circumferential feature or interacting dent causes a prominent peak near nominal radius at the deflection boundary, and the global maximum is the most significant marker. Disable for standard dent detection where sustained radius/slope deviation marks the start. Inspect data plots to confirm a prominent peak near nominal radius.
    
    Returns
    -------
    start_idx : int or None
        The index where deflection initiates (with buffer applied), or None if no deflection detected.
    start_val : float or None
        The radius at start_idx, or None if no deflection detected.
    """
    if len(data) < window_size:
        raise ValueError("Data length must be at least the window size.")
    
    # Compute smoothed first derivative
    der = savgol_filter(data, window_length=window_size, polyorder=2, deriv=1)
    
    # Closeness tolerance for radius
    closeness_tol = (closeness_percentage / 100) * nominal_radius
    
    # Define pristine condition: flat slope and radius close to nominal
    def is_pristine(i):
        return abs(der[i]) <= slope_tolerance and abs(data[i] - nominal_radius) <= closeness_tol
    
    start_idx = 0
    if circumferential_mode:
        # Find global maximum among points with near-zero slope and radius close to nominal
        max_radius = -np.inf
        max_idx = None
        i = 0
        while i < len(der):
            if is_pristine(i):
                # Check for consecutive points to confirm the maximum
                consec = 1
                j = i + 1
                while j < len(der) and is_pristine(j):
                    consec += 1
                    j += 1
                if consec >= min_consecutive_deviations:
                    # Check if this region contains a higher radius
                    region_max = np.max(data[i:j])
                    if region_max > max_radius:
                        max_radius = region_max
                        max_idx = i + np.argmax(data[i:j])
                i = j
            else:
                i += 1
        start_idx = max_idx
    else:
        # Default mode: find sustained deviation from pristine
        i = 0
        while i < len(der):
            if is_pristine(i):
                start_idx = i + 1
                i += 1
            else:
                # Check for consecutive deviations
                consec = 1
                j = i + 1
                while j < len(der) and not is_pristine(j):
                    consec += 1
                    j += 1
                if consec >= min_consecutive_deviations:
                    break
                else:
                    # Skip the noise blip and continue
                    start_idx = j
                    i = j
    
    # If no valid deflection point found (entire data pristine or no valid maximum)
    if start_idx >= len(data):
        return None, None
    
    # Apply buffer offset (subtract for initiation to include more preceding data)
    start_idx = max(0, start_idx - buffer_offset)

    # Cap to data length
    start_val = data[start_idx] if start_idx < len(data) else None

    return start_idx, start_val

def get_restraint_parameter(AAX_15: float, ATR_15: float, LTR_70: float, LAX_15: float, LAX_30: float, LAX_50: float, LTR_80: float) -> float:
    """
    Calculate the Restraint Parameter (RP) based on the characteristic lengths and areas for the dent quadrant.

    Parameters
    ----------
    AAX_15 : float
        Axial Area at 15% dent depth.
    ATR_15 : float
        Circumferential Area at 15% dent depth.
    LTR_70 : float
        Circumferential Length at 70% dent depth.
    LAX_15 : float
        Axial Length at 15% dent depth.
    LAX_30 : float
        Axial Length at 30% dent depth.
    LAX_50 : float
        Axial Length at 50% dent depth.
    LTR_80 : float
        Circumferential Length at 80% dent depth.

    Returns
    -------
    rp : float
        The calculated Restraint Parameter score.
    """
    rp = max(18 * abs(AAX_15 - ATR_15) ** (1/2) / LTR_70, 8 * (LAX_15 / LAX_30) ** (1/4) * ((LAX_30 - LAX_50) / LTR_80) ** (1/2))
    return rp

class DentProfiles:
    def __init__(self, 
                 df: pd.DataFrame, 
                 OD: float, 
                 WT: float, 
                 ignore_edge: float = 0.1,
                 percentages_axial: list[int] = [95, 90, 85, 75, 60, 50, 40, 30, 20, 15, 10, 5],
                 percentages_circ: list[int] = [90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 15, 10],
                 percentages_area: list[int] = [85, 75, 60, 50, 40, 30, 20, 15, 10],
                 process_data: bool = True,
                 file_path: str = None
                 ):
        """
        Initialize the DentProfiles class.
        """
        self._results_axial_us = None
        self._results_axial_ds = None
        self._results_circ_us_ccw = None
        self._results_circ_us_cw = None
        self._results_circ_ds_ccw = None
        self._results_circ_ds_cw = None
        if process_data:
            self._process_data(df, OD, WT, ignore_edge, percentages_axial, percentages_circ, percentages_area, file_path)

    def __repr__(self):
        # Provide a concise summary of key attributes. Combine the dictionaries into a single table for display.
        # The table will have the quadrant names as columns, the percentage levels as the index, and the lengths and areas as values.
        columns = ["US-Lengths", "US-Areas", "DS-Lengths", "DS-Areas", "US-CCW-Lengths", "US-CCW-Areas", "US-CW-Lengths", "US-CW-Areas", "DS-CCW-Lengths", "DS-CCW-Areas", "DS-CW-Lengths", "DS-CW-Areas"]
        # There are some repeating percentage levels, so we will create a unique sorted list among all the lengths and areas
        # Convert all index_values to integers
        index_values = sorted(set(list(self._results_axial_us["lengths"].keys()) + 
                                        list(self._results_axial_us["areas"].keys()) + 
                                        list(self._results_axial_ds["lengths"].keys()) + 
                                        list(self._results_axial_ds["areas"].keys()) + 
                                        list(self._results_circ_us_ccw["lengths"].keys()) +
                                        list(self._results_circ_us_ccw["areas"].keys()) +
                                        list(self._results_circ_us_cw["lengths"].keys()) +
                                        list(self._results_circ_us_cw["areas"].keys()) +
                                        list(self._results_circ_ds_ccw["lengths"].keys()) +
                                        list(self._results_circ_ds_ccw["areas"].keys()) +
                                        list(self._results_circ_ds_cw["lengths"].keys()) +
                                        list(self._results_circ_ds_cw["areas"].keys())),
                                        reverse=True)
        index_values = [int(i) for i in index_values]
        # Go through each index value and extract the corresponding lengths and areas from each quadrant, using NaN if not present
        data = []
        for idx in index_values:
            row = {
                columns[0]: self._results_axial_us["lengths"].get(idx, "")["length"] if self._results_axial_us["lengths"].get(idx, "") != "" else "",
                columns[1]: self._results_axial_us["areas"].get(idx, ""),
                columns[2]: self._results_axial_ds["lengths"].get(idx, "")["length"] if self._results_axial_ds["lengths"].get(idx, "") != "" else "",
                columns[3]: self._results_axial_ds["areas"].get(idx, ""),
                columns[4]: self._results_circ_us_ccw["lengths"].get(idx, "")["length"] if self._results_circ_us_ccw["lengths"].get(idx, "") != "" else "",
                columns[5]: self._results_circ_us_ccw["areas"].get(idx, ""),
                columns[6]: self._results_circ_us_cw["lengths"].get(idx, "")["length"] if self._results_circ_us_cw["lengths"].get(idx, "") != "" else "",
                columns[7]: self._results_circ_us_cw["areas"].get(idx, ""),
                columns[8]: self._results_circ_ds_ccw["lengths"].get(idx, "")["length"] if self._results_circ_ds_ccw["lengths"].get(idx, "") != "" else "",
                columns[9]: self._results_circ_ds_ccw["areas"].get(idx, ""),
                columns[10]: self._results_circ_ds_cw["lengths"].get(idx, "")["length"] if self._results_circ_ds_cw["lengths"].get(idx, "") != "" else "",
                columns[11]: self._results_circ_ds_cw["areas"].get(idx, ""),
            }
            data.append(row)
        
        summary_df = pd.DataFrame(data=data, index=index_values, columns=columns)
        return_string = (
            f"DentProfiles Summary:\n"
            f" - OD: {round(self._OD, 2)}-inch, WT: {round(self._WT, 3)}-inch\n"
            f" - Dent Depth based on overall radius of {round(self._nominal_radius, 3)}-inch: {round(self._dent_depth, 3)}-inch ({round(self._dent_depth_percent, 3)} %OD)\n"
            f" - Dent Location and Radius: (Axial = {round(self._axial_min, 2)}-inch, Circumferential = {round(self._circ_min, 2)}-deg, Radius = {round(self._radius_min, 2)}-inch)\n"
            # Insert the Summary Table here
            f"\n"
            f"Summary of Lengths and Areas by Quadrant (% levels as index):\n"
            f"{summary_df.to_string()}\n"
        )
        return return_string

    def _process_data(self, 
                      df: pd.DataFrame, 
                      OD: float, 
                      WT: float, 
                      ignore_edge: float,
                      percentages_axial: list,
                      percentages_circ: list,
                      percentages_area: list, 
                      file_path: str):
        """
        Class to handle dent profile data and compute key metrics.
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing dent contour with df.index is 'Axial Displacement', df.columns is 'Circumferential Orientation', and df.values is 'Radius'.
        ignore_edge : float, optional (default=0.1)
            Fraction of data to ignore at each edge when determining nominal radius.
        """
        self._df = df
        self._OD = OD
        self._WT = WT
        # Locate the deepest point, using the ignore_edge parameter to avoid edge effects
        start_idx = math.ceil(df.shape[0]*ignore_edge)
        end_idx = math.floor(df.shape[0]*(1-ignore_edge))
        df_trim = df.iloc[start_idx:end_idx, :]
        self._axial_min, self._circ_min = df_trim.stack().idxmin()
        self._radius_min = df_trim.loc[self._axial_min, self._circ_min]
        # Extract the Axial and Circumferential profiles at the deepest point
        self._axial_profile = self._df[self._circ_min]
        self._circ_profile = self._df.loc[self._axial_min]
        # Split Axial data into US/DS
        self._axial_us = self._axial_profile.loc[:self._axial_min]
        self._axial_ds = self._axial_profile.loc[self._axial_min:]
        # Split Circumferential data into CCW/CW
        self._circ_ccw = self._circ_profile.loc[:self._circ_min]
        self._circ_cw = self._circ_profile.loc[self._circ_min:]
        # Determine the nominal internal radius
        self._nominal_radius = self.get_nominal(expected_nominal=(OD/2 - WT), ignore_edge=ignore_edge)
        self._dent_depth = self._nominal_radius - self._radius_min
        self._dent_depth_percent = (self._dent_depth / OD) * 100
        # Determine the baseline index and radii for all four quadrants (index, radius)
        self._baseline_us = self.get_baseline(self._axial_us, axial_circ="axial")
        self._baseline_ds = self.get_baseline(self._axial_ds, axial_circ="axial", outbound_data=True, slope_tolerance=0.004)
        # The Circumferential baselines will use the US and DS axial baselines. But will need to find the index in the circumferential profile
        self._baseline_us_ccw = self.get_baseline_circ(self._circ_ccw, self._baseline_us[2])
        self._baseline_us_cw = self.get_baseline_circ(self._circ_cw, self._baseline_us[2], outbound_data=True)
        self._baseline_ds_ccw = self.get_baseline_circ(self._circ_ccw, self._baseline_ds[2])
        self._baseline_ds_cw = self.get_baseline_circ(self._circ_cw, self._baseline_ds[2], outbound_data=True)
        # Re-establish the dent depth value for each quadrant
        self._dent_depth_us = self._baseline_us[2] - self._radius_min
        self._dent_depth_ds = self._baseline_ds[2] - self._radius_min
        self._dent_depth_us_ccw = self._baseline_us_ccw[2] - self._radius_min
        self._dent_depth_us_cw = self._baseline_us_cw[2] - self._radius_min
        self._dent_depth_ds_ccw = self._baseline_ds_ccw[2] - self._radius_min
        self._dent_depth_ds_cw = self._baseline_ds_cw[2] - self._radius_min
        # Iterate through all four quadrants to determine lengths and areas
        self._results_axial_us = self.get_measurements(self._axial_us, self._dent_depth_us, self._axial_min, self._baseline_us, percentages_axial, percentages_area)
        self._results_axial_ds = self.get_measurements(self._axial_ds, self._dent_depth_ds, self._axial_min, self._baseline_ds, percentages_axial, percentages_area, outbound_data=True)
        self._results_circ_us_ccw = self.get_measurements(self._circ_ccw, self._dent_depth_us_ccw, self._circ_min, self._baseline_us_ccw, percentages_circ, percentages_area)
        self._results_circ_us_cw = self.get_measurements(self._circ_cw, self._dent_depth_us_cw, self._circ_min, self._baseline_us_cw, percentages_circ, percentages_area, outbound_data=True)
        self._results_circ_ds_ccw = self.get_measurements(self._circ_ccw, self._dent_depth_ds_ccw, self._circ_min, self._baseline_ds_ccw, percentages_circ, percentages_area)
        self._results_circ_ds_cw = self.get_measurements(self._circ_cw, self._dent_depth_ds_cw, self._circ_min, self._baseline_ds_cw, percentages_circ, percentages_area, outbound_data=True)
        # Create three figures
        if file_path is not None:
            self.create_figure("Axial", self._axial_us, self._axial_ds, self._results_axial_us, self._results_axial_ds, self._axial_min, file_path)
            self.create_figure("Circ_US", self._circ_ccw, self._circ_cw, self._results_circ_us_ccw, self._results_circ_us_cw, self._circ_min, file_path)
            self.create_figure("Circ_DS", self._circ_ccw, self._circ_cw, self._results_circ_ds_ccw, self._results_circ_ds_cw, self._circ_min, file_path)
    
    def graph(self, quadrant: str) -> plt.Figure:
        """Generate and return a matplotlib Figure for the specified quadrant ('Axial', 'Circ_US', 'Circ_DS')."""
        if quadrant == "Axial":
            fig = self.create_figure("Axial", self._axial_us, self._axial_ds, self._results_axial_us, self._results_axial_ds, self._axial_min)
        elif quadrant == "Circ_US":
            fig = self.create_figure("Circ_US", self._circ_ccw, self._circ_cw, self._results_circ_us_ccw, self._results_circ_us_cw, self._circ_min)
        elif quadrant == "Circ_DS":
            fig = self.create_figure("Circ_DS", self._circ_ccw, self._circ_cw, self._results_circ_ds_ccw, self._results_circ_ds_cw, self._circ_min)
        else:
            raise ValueError("Invalid quadrant specified. Choose from 'Axial', 'Circ_US', 'Circ_DS'.")
        return fig
    
    @property
    def min_idx(self) -> tuple[int, int]:
        """Tuple of (Axial index, Circumferential index) of the deepest point."""
        axial_idx = self._df.index.get_loc(self._axial_min)
        circ_idx = self._df.columns.get_loc(self._circ_min)
        return (axial_idx, circ_idx)
    @property
    def df(self) -> pd.DataFrame:
        """DataFrame of the dent contour."""
        return self._df
    @property
    def axial_profile(self) -> pd.Series:
        """Axial profile at the deepest point."""
        return self._axial_profile
    @property
    def circ_profile(self) -> pd.Series:
        """Circumferential profile at the deepest point."""
        return self._circ_profile
    @property
    def axial_us(self) -> pd.Series:
        """Axial profile upstream of the deepest point."""
        return self._axial_us
    @property
    def axial_ds(self) -> pd.Series:
        """Axial profile downstream of the deepest point."""
        return self._axial_ds
    @property
    def circ_ccw(self) -> pd.Series:
        """Circumferential profile counter-clockwise of the deepest point."""
        return self._circ_ccw
    @property
    def circ_cw(self) -> pd.Series:
        """Circumferential profile clockwise of the deepest point."""
        return self._circ_cw
    @property
    def axial_min(self) -> float:
        """Axial location of the deepest point."""
        return self._axial_min
    @property
    def circ_min(self) -> float:
        """Circumferential location of the deepest point."""
        return self._circ_min
    @property
    def depth(self) -> float:
        """Depth of the dent (nominal radius - minimum radius)."""
        return self._dent_depth
    @property
    def nominal_radius(self) -> float:
        """Nominal internal radius."""
        return self._nominal_radius
    @property
    def baseline_us(self) -> tuple[int, float]:
        """Baseline radius upstream of the deepest point."""
        return self._baseline_us
    @property
    def baseline_ds(self) -> tuple[int, float]:
        """Baseline radius downstream of the deepest point."""
        return self._baseline_ds
    @property
    def baseline_us_ccw(self) -> tuple[int, float]:
        """Baseline radius counter-clockwise of the deepest point."""
        return self._baseline_us_ccw
    @property
    def baseline_us_cw(self) -> tuple[int, float]:
        """Baseline radius clockwise of the deepest point."""
        return self._baseline_us_cw
    @property
    def baseline_ds_ccw(self) -> tuple[int, float]:
        """Baseline radius counter-clockwise of the deepest point."""
        return self._baseline_ds_ccw
    @property
    def baseline_ds_cw(self) -> tuple[int, float]:
        """Baseline radius clockwise of the deepest point."""
        return self._baseline_ds_cw
    @property
    def US_LAX(self) -> list[float]:
        """US Axial Lengths for all percentages."""
        temp_dict = self._results_axial_us["lengths"]
        output_list = [val["length"] for val in temp_dict.values()]
        return output_list
    @property
    def US_AAX(self) -> list[float]:
        """US Axial Areas for all percentages."""
        return list(self._results_axial_us["areas"].values())
    @property
    def DS_LAX(self) -> list[float]:
        """DS Axial Lengths for all percentages."""
        temp_dict = self._results_axial_ds["lengths"]
        output_list = [val["length"] for val in temp_dict.values()]
        return output_list
    @property
    def DS_AAX(self) -> list[float]:
        """DS Axial Areas for all percentages."""
        return list(self._results_axial_ds["areas"].values())
    @property
    def US_CCW_LTR(self) -> list[float]:
        """US Circumferential CCW Lengths for all percentages."""
        temp_dict = self._results_circ_us_ccw["lengths"]
        output_list = [val["length"] for val in temp_dict.values()]
        return output_list
    @property
    def US_CCW_ATR(self) -> list[float]:
        """US Circumferential CCW Areas for all percentages."""
        return list(self._results_circ_us_ccw["areas"].values())
    @property
    def US_CW_LTR(self) -> list[float]:
        """US Circumferential CW Lengths for all percentages."""
        temp_dict = self._results_circ_us_cw["lengths"]
        output_list = [val["length"] for val in temp_dict.values()]
        return output_list
    @property
    def US_CW_ATR(self) -> list[float]:
        """US Circumferential CW Areas for all percentages."""
        return list(self._results_circ_us_cw["areas"].values())
    @property
    def DS_CCW_LTR(self) -> list[float]:
        """DS Circumferential CCW Lengths for all percentages."""
        temp_dict = self._results_circ_ds_ccw["lengths"]
        output_list = [val["length"] for val in temp_dict.values()]
        return output_list
    @property
    def DS_CCW_ATR(self) -> list[float]:
        """DS Circumferential CCW Areas for all percentages."""
        return list(self._results_circ_ds_ccw["areas"].values())
    @property
    def DS_CW_LTR(self) -> list[float]:
        """DS Circumferential CW Lengths for all percentages."""
        temp_dict = self._results_circ_ds_cw["lengths"]
        output_list = [val["length"] for val in temp_dict.values()]
        return output_list
    @property
    def DS_CW_ATR(self) -> list[float]:
        """DS Circumferential CW Areas for all percentages."""
        return list(self._results_circ_ds_cw["areas"].values())

    def get_nominal(self, expected_nominal: float, threshold: float = 0.05, ignore_edge: float = 0.1) -> float:
        """
        Determine the nominal radius from the profile data.

        Parameters
        ----------
        expected_nominal : float, optional
            Expected nominal radius to validate against.
        threshold : float, optional (default=0.05)
            Maximum allowed deviation from expected_nominal, as a fraction of the expected value.
        ignore_edge : float, optional (default=0.1)
            Fraction of data to ignore at each edge when determining nominal radius.
        Returns
        -------
        nominal_radius : float
            The determined nominal radius.
        """
        # Use the outer 10% of the data to determine nominal radius
        n_points = max(1, math.ceil(self._df.shape[0]*ignore_edge))
        edge_data = pd.concat([self._df.iloc[:n_points, :], self._df.iloc[-n_points:, :]])
        nominal_radius = edge_data.stack().mean()
        if expected_nominal is not None and abs(nominal_radius - expected_nominal) <= (expected_nominal * threshold):
            # If the expected nominal is provided and close enough, use it
            nominal_radius = expected_nominal
        return nominal_radius

    def get_baseline(self, 
                     data: pd.Series, 
                     axial_circ: str = "axial", 
                     axial_default: float = 0.025, 
                     circ_default: float = -0.15,
                     window_size: int = 11,
                     slope_tolerance: float = 0.003,
                     closeness_percentage: float = 5,
                     buffer_offset: float = 5,
                     min_consecutive_deviations: int = 3,
                     circumferential_mode: bool = False,
                     outbound_data: bool = False) -> tuple[int, float]:
        """
        Determine the baseline radius which will be the reference line for all calculations. This can either be a fixed
        offset from the nominal radius or determined from the changing slope in the profile.
        
        Parameters
        ----------
        data : pd.Series
            The profile data (axial or circumferential) to analyze for baseline determination.
        axial_circ : str
            Specifies whether to use the axial or circumferential profile for baseline determination.
        axial_default : float
            Default value for the axial profile baseline (default is 2.5%).
        circ_default : float
            Default value for the circumferential profile baseline (default is -15%).
        window_size : int, optional (default=11)
            Window for Savitzky-Golay filter to smooth derivatives.
        slope_tolerance : float, optional (default=0.003)
            Slope threshold for pristine sections or local maxima.
        closeness_percentage : float, optional (default=5)
            Percentage tolerance for radius closeness to nominal.
        buffer_offset : int, optional (default=5)
            Points to subtract from start index for preceding data.
        min_consecutive_deviations : int, optional (default=3)
            Minimum consecutive deviations (default mode) or consecutive points at local maximum (circumferential mode).
        circumferential_mode : bool, optional (default=False)
            If True, detects deflection at the first local maximum (slope ≈ 0) near nominal radius. If False, uses sustained deviation.
        outbound_data : bool, optional (default=False)
            If False, includes inbound data points for baseline determination. If True, reverses the data to treat it as inbound.
            - Guidance: Ensure data direction matches this flag. Reverse if necessary.

        Returns
        -------
        baseline_index : int
            The index in the profile corresponding to the baseline radius.
        baseline_axial : float
            The axial location corresponding to the baseline radius.
        baseline_radius : float
            The determined baseline radius.
        """
        # Default baseline radius if no suitable point is found
        if axial_circ.lower() == "axial":
            baseline_default = self._nominal_radius - axial_default * self._dent_depth
        elif axial_circ.lower() == "circumferential":
            baseline_default = self._nominal_radius - circ_default * self._dent_depth
        else:
            # If invalid option, default to axial method
            baseline_default = self._nominal_radius - axial_default * self._dent_depth

        # Calculate the baseline radius from the profile data
        baseline_index, baseline_axial, baseline_val = find_deflection(data, self._nominal_radius, window_size=window_size, slope_tolerance=slope_tolerance, closeness_percentage=closeness_percentage, buffer_offset=buffer_offset, min_consecutive_deviations=min_consecutive_deviations, circumferential_mode=circumferential_mode, outbound_data=outbound_data)

        if baseline_index is not None and baseline_val is not None:
            return baseline_index, baseline_axial, baseline_val
        else:
            # If no valid baseline found, return default and find the closest point in data to the default,
            closest_idx = (data - baseline_default).abs().idxmin()
            closest_axial = data.index[closest_idx] if closest_idx is not None else None
            return closest_idx, closest_axial, baseline_default

    def get_baseline_circ(self, 
                          data: pd.Series, 
                          baseline_axial_radius: float, 
                          outbound_data: bool = False) -> tuple[int, float]:
        """
        Determine the baseline radius in the circumferential profile using the axial baseline index.

        Parameters
        ----------
        baseline_axial_radius : float
            The baseline radius from the axial profile.
        outbound_data : bool, optional (default=False)
            If False, includes inbound data points for baseline determination. If True, reverses the data to treat it as inbound.
            - Guidance: Ensure data direction matches this flag. Reverse if necessary.

        Returns
        -------
        baseline_index : int
            The index in the circumferential profile corresponding to the baseline radius.
        baseline_deg : float
            The circumferential location corresponding to the baseline radius.
        baseline_radius : float
            The determined baseline radius in the circumferential profile.
        """
        # Find the circumferential index that has the same radius as the axial baseline
        if not outbound_data:
            data = data.iloc[::-1]

        # Find the index where the profile crosses the baseline radius (choose index after).
        # Since there can be repeating radial values, we find the index closest to the minimum (dent depth).
        # Find the first index where the profile crosses the target radius (choose index after).
        crossing_idx = data[data >= baseline_axial_radius].first_valid_index()
        # Interpolate between the crossing index and the previous index to get a more accurate length
        if crossing_idx is None or crossing_idx == data.index[0]:
            # If no crossing found, use the last index
            circ_index = len(data) - 1
            circ_deg = data.index[circ_index]
            circ_radius = data.loc[circ_deg]
        else:
            # Linear interpolation to find the exact crossing point
            circ_index = data.index.get_loc(crossing_idx)
            x0, x1 = data.index[data.index.get_loc(crossing_idx) - 1], crossing_idx
            y0, y1 = data.loc[x0], data.loc[x1]
            if y1 != y0:
                circ_deg = x0 + (baseline_axial_radius - y0) * (x1 - x0) / (y1 - y0)
            else:
                circ_deg = x1  # If y1 == y0, just take the crossing index
            circ_radius = baseline_axial_radius

        return circ_index, circ_deg, circ_radius
    
    def get_measurements(self, data: pd.Series, dent_depth: float, dent_location: float, baseline: tuple, percentages_length: list, percentages_area: list, outbound_data: bool = False) -> dict:
        """
        Calculate the lengths and areas of the dent in all four quadrants using the specified percentages of the dent depth.

        To minimize error and select the correct locations, the lengths are determined by finding the points in each quadrant where the profile crosses the target radius
        defined by the specified percentage of the dent depth.

        The areas are calculated using the trapezoidal rule between the baseline and the profile, from the minimum point to the length point.

        Parameters
        ----------
        data : pd.Series
            The profile data (axial or circumferential) to analyze for length determination.
        dent_depth : float
            The dent depth for the profile.
        dent_location : float
            The axial or circumferential location of the dent minimum for the profile.
        baseline : tuple
            The baseline (index, axial location, corresponding radius) for the profile.
        percentages_length : list
            List of percentages to calculate the lengths at.
        percentages_area : list
            List of percentages to calculate the areas at.
        outbound_data : bool, optional (default=False)
            If False, includes inbound data points for length determination. If True, reverses the data to treat it as inbound.
            - Guidance: Ensure data direction matches this flag. Reverse if necessary.

        Returns
        -------
        lengths : dict
            Dictionary with lengths and corresponding starting axial location and radius in each quadrant.
        areas : dict
            Dictionary with areas in each quadrant.
        """
        if not outbound_data:
            # For inbound data, we reverse the data and search from the minimum to the target
            data = data.iloc[::-1]
        
        lengths = {}
        for pct in percentages_length:
            target_radius = self._radius_min + (1 - pct / 100) * dent_depth
            # Find the index where the profile crosses the target radius (choose index after).
            # Since there can be repeating radial values, we find the index closest to the minimum (dent depth).
            # Find the first index where the profile crosses the target radius (choose index after).
            crossing_idx = data[data >= target_radius].first_valid_index()
            # Interpolate between the crossing index and the previous index to get a more accurate length
            if crossing_idx is None or crossing_idx == data.index[0]:
                # If no crossing found or crossing is at the start, use the default length
                length = None
                interp_position = None
            else:
                # Linear interpolation to find the exact crossing point
                x0, x1 = data.index[data.index.get_loc(crossing_idx) - 1], crossing_idx
                y0, y1 = data.loc[x0], data.loc[x1]
                if y1 != y0:
                    interp_position = x0 + (target_radius - y0) * (x1 - x0) / (y1 - y0)
                else:
                    interp_position = x1  # If y1 == y0, just take the crossing index

                # Calculate length from the minimum index to the interpolated index
                length = abs(interp_position - dent_location)
            # Store the length
            lengths[pct] = {"length": length, "position": interp_position, "radius": target_radius}

        # Calculate areas using the trapezoidal rule for ALL data points starting from the minimum to the baseline
        areas = []
        # Warning: the baseline index is based on the original data, so we need to adjust
        if not outbound_data:
            data_to_use = data[(data.index >= baseline[1])]

            for i in range(len(data_to_use) - 1, 0, -1):
                # Trapezoidal area between two points
                axial_i, axial_im1 = data_to_use.index[i - 1], data_to_use.index[i]
                rad_i, rad_im1 = data_to_use.loc[axial_i], data_to_use.loc[axial_im1]
                trap_area = 0.5 * (axial_i - axial_im1) * abs((rad_i - baseline[2]) + (rad_im1 - baseline[2]))
                areas.append((axial_i, trap_area))
        else:
            data_to_use = data[(data.index <= baseline[1])]

            for i in range(1, len(data_to_use)):
                # Trapezoidal area between two points
                axial_i, axial_im1 = data_to_use.index[i], data_to_use.index[i - 1]
                rad_i, rad_im1 = data_to_use.loc[axial_i], data_to_use.loc[axial_im1]
                trap_area = 0.5 * (axial_i - axial_im1) * abs((rad_i - baseline[2]) + (rad_im1 - baseline[2]))
                areas.append((axial_i, trap_area))

        # Calculate the cumulative area
        cum_areas = {}
        for pct in percentages_area:
            # Use a generator expression to find the value of v["position"] in the lengths.items() list where the percentage (k) matches the given value pct
            axial_at_pct = next((v["position"] for k, v in lengths.items() if k == pct), None)
            idx_at_pct = min(range(len(areas)), key=lambda i: abs(areas[i][0] - axial_at_pct)) if axial_at_pct is not None else None
            # Store the cumulative area from the minimum to the length point
            if not outbound_data:
                cum_areas[pct] = sum(area for _, area in areas[idx_at_pct:]) if idx_at_pct is not None else None
            else:
                cum_areas[pct] = sum(area for _, area in areas[:idx_at_pct + 1]) if idx_at_pct is not None else None

        return {"lengths": lengths, "areas": cum_areas}

    def create_figure(self, quadrant: str, profile_us: pd.Series, profile_ds: pd.Series, results_us: dict, results_ds: dict, dent_location: float, file_path: str=None, palette: list = [
            '#1f77b4',  # muted blue
            '#ff7f0e',  # safety orange
            '#2ca02c',  # cooked asparagus green
            '#d62728',  # brick red
            '#9467bd',  # muted purple
            '#8c564b',  # chestnut brown
            '#e377c2',  # raspberry yogurt pink
            '#7f7f7f',  # middle gray
            '#bcbd22',  # curry yellow‑green
            '#17becf',  # blue‑teal
            '#aec7e8',  # light pastel blue
            '#ffbb78'   # light pastel orange
        ]):
        """
        Create a matplotlib figure showing the dent profile and the lengths plotted at each percentage.
        Use the indicated baseline for the profile and dent profile segment.
        """
        # Check the corresponding quadrant for custom text labels
        if quadrant.lower() == "axial":
            title = "Axial Lengths"
            xlabel = "Axial Distance (in)"
            ylabel = "Radius (in)"
            us_label = "Upstream"
            ds_label = "Downstream"
            data_label = "LAX"
            data_label2 = ["US","DS"]
        if quadrant.lower() == "circ_us":
            title = "Upstream Circumferential Lengths"
            xlabel = "Circumferential Distance (deg)"
            ylabel = "Radius (in)"
            us_label = "Counterclockwise"
            ds_label = "Clockwise"
            data_label = "LTR"
            data_label2 = ["US-CCW","DS-CW"]
        if quadrant.lower() == "circ_ds":
            title = "Downstream Circumferential Lengths"
            xlabel = "Circumferential Distance (deg)"
            ylabel = "Radius (in)"
            us_label = "Counterclockwise"
            ds_label = "Clockwise"
            data_label = "LTR"
            data_label2 = ["US-CCW","DS-CW"]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(profile_us.index, profile_us, label=us_label, color=palette[0])
        ax.plot(profile_ds.index, profile_ds, label=ds_label, color=palette[1], linestyle='--')

        for i, (p, vals) in enumerate(results_us["lengths"].items()):
            ax.plot([vals["position"], dent_location], [vals["radius"], vals["radius"]], color=palette[(2+i)%len(palette)], linestyle='-', linewidth=1, label=f'{data_label}{p}% {data_label2[0]}')

        for i, (p, vals) in enumerate(results_ds["lengths"].items()):
            ax.plot([dent_location, vals["position"]], [vals["radius"], vals["radius"]], color=palette[(2+i)%len(palette)], linestyle='--', linewidth=1, label=f'{data_label}{p}% {data_label2[1]}')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        fig.tight_layout()
        if file_path:
            fig.savefig(str(file_path).replace('.xlsx', f'_{quadrant}_Lengths.png'), dpi=300)
            plt.close(fig)
        
