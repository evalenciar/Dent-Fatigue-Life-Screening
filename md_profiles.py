"""
Determine and evaluate the dent profile for all quadrants according to 
API RP 1183 Section 6.2 Dent Geometry Profile Characterization
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def find_deflection(data: pd.Series, 
                    initial_guess: float = 0.5,
                    angle_threshold: float = 0.05,
                    plot_lines: bool = False) -> tuple[int | None, float | None, float | None]:
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
    initial_guess : float, optional (default=0.5)
        Initial guess for the shoulder location as a fraction of the data length.
    angle_threshold : float, optional (default=0.05)
        Threshold angle in radians below which the shoulder is not considered valid.
    plot_lines : bool, optional (default=False)
        Whether to plot the linear fits and residuals for debugging.
    
    Returns
    -------
    init_idx : int or None
        The relative index in data where deflection starts (with buffer), or None if not detected.
    init_axial : float or None
        The axial location at init_idx, or None if not detected.
    init_radius : float or None
        The radius at init_idx, or None if not detected.
    """
    x = data.index.to_numpy()
    y = data.to_numpy()
    init_idx, init_axial, init_radius = _find_shoulder_min(x, y, initial_guess, angle_threshold, plot_lines)
    if init_idx is None or init_radius is None or init_axial is None:
        return None, None, None
    
    return init_idx, init_axial, init_radius
    
def _find_shoulder_min(x: np.ndarray,
                       y: np.ndarray,
                       initial_guess: float = 0.5,
                       angle_threshold: float = 0.05,
                       plot_lines: bool = False) -> tuple[int | None, float | None, float | None]:
    """
    Internal helper to find the dent shoulder location using minimization of residuals from two linear fits.

    Parameters
    ----------
    data : np.ndarray
        Array of caliper radius measurements along the pipeline section. Must be a 1D array of floating-point values representing radii at sequential points.
        - Guidance: Ensure the data is preprocessed to represent either the inbound (pristine to dent) or reversed outbound (dent to pristine) half. Array length should be sufficient to capture the shoulder region. Smooth extreme outliers manually if necessary.
        - Example: `np.array([10.0, 9.99, 9.98, ..., 8.5])` where 10.0 is nominal radius.
    initial_guess : float, optional
        Initial guess for the breakpoint as a fraction of the data length (where 0 < x < 1). Default is 0.5.
        - Guidance: Choose a value that roughly estimates the shoulder location. For example, 0.5 for mid-point, or adjust based on visual inspection of the data.
    angle_threshold : float, optional
        Threshold angle in degrees below which the shoulder is not considered valid. Default is 0.05 degrees.
    plot_lines : bool, optional
        Whether to plot the fitted lines and breakpoint for visualization. Default is False.
    Returns
    -------
    tuple[int | None, float | None, float | None]
        A tuple containing the index of the shoulder, the axial position, and the radius at the shoulder. Returns (None, None, None) if not found.
    """
    from scipy.optimize import minimize
    def _minimize_residuals(breakpoint_frac, x, y):
        breakpoint_idx = int(breakpoint_frac * (len(x) - 1))

        # Fit first line to data before the breakpoint
        coeffs1 = np.polyfit(x[:breakpoint_idx + 1], y[:breakpoint_idx + 1], 1)
        line1 = np.polyval(coeffs1, x[:breakpoint_idx + 1])

        # Fit second line to data after the breakpoint
        coeffs2 = np.polyfit(x[breakpoint_idx:], y[breakpoint_idx:], 1)
        line2 = np.polyval(coeffs2, x[breakpoint_idx:])

        # Calculate residuals
        residuals1 = y[:breakpoint_idx + 1] - line1
        residuals2 = y[breakpoint_idx:] - line2

        total_residual = np.sum(residuals1**2) + np.sum(residuals2**2)
        return total_residual
    
    def _find_closest_point(breakpoint_frac, x, y, plot_lines=False) -> tuple[int | None, float | None, float | None]:
        breakpoint_idx = int(breakpoint_frac * (len(x) - 1))
        if breakpoint_idx <= 0 or breakpoint_idx >= len(x) - 1:
            return None, None, None  # Invalid breakpoint

        # Fit first line to data before the breakpoint
        coeffs1 = np.polyfit(x[:breakpoint_idx + 1], y[:breakpoint_idx + 1], 1)
        line1 = np.polyval(coeffs1, x[:breakpoint_idx + 1])

        # Fit second line to data after the breakpoint
        coeffs2 = np.polyfit(x[breakpoint_idx:], y[breakpoint_idx:], 1)
        line2 = np.polyval(coeffs2, x[breakpoint_idx:])

        # Find the intersection point of the two lines
        if coeffs1[0] == coeffs2[0]:
            return None, None, None  # Parallel lines, no intersection
        A = np.array([[coeffs1[0], -1], [coeffs2[0], -1]])
        b = np.array([-coeffs1[1], -coeffs2[1]])
        intersection = np.linalg.solve(A, b)

        # Determine the angle between the two lines. If it is below a certain threshold, return None
        # angle = np.arccos((1 + coeffs1[0] * coeffs2[0]) / (np.sqrt(1 + coeffs1[0]**2) * np.sqrt(1 + coeffs2[0]**2)))
        angle = np.arctan(abs((coeffs2[0] - coeffs1[0]) / (1 + coeffs1[0] * coeffs2[0])))
        if angle < np.deg2rad(angle_threshold):  # Threshold of 5 degrees
            return None, None, None

        # Find the closest data point to the intersection
        distances = np.sqrt((x - intersection[0])**2 + (y - intersection[1])**2)
        closest_idx = int(np.argmin(distances))

        if plot_lines:
            plt.figure()
            plt.scatter(x, y, c='b', label='Data', s=0.5)
            plt.plot(x[:breakpoint_idx + 1], line1, 'r-', label='Fit 1')
            plt.plot(x[breakpoint_idx:], line2, 'g-', label='Fit 2')
            # plt.scatter(intersection[0], intersection[1], c='k', label='Intersection')
            plt.scatter(x[closest_idx], y[closest_idx], c='orange', marker='d', label="Dent Shoulder")
            plt.legend()
            plt.xlabel('Axial Position [in]')
            plt.ylabel('Radius [in]')
            plt.title('Dent Shoulder Detection via Two-Line Fit')
            plt.show()

        return closest_idx, x[closest_idx], y[closest_idx]
    
    if initial_guess <= 0:
        initial_guess = 0.01
    elif initial_guess >= 1:
        initial_guess = 0.99
    
    result = minimize(_minimize_residuals, 
                      initial_guess, 
                      args=(x, y), method='Nelder-Mead', 
                      options={'xatol': 1e-8, 'fatol': 1e-8},
                      bounds=[(0.01, 0.99)])
    best_breakpoint = result.x[0]
    return _find_closest_point(best_breakpoint, x, y, plot_lines)

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
                 equal_baseline: bool = False,
                 ignore_edge: float = 0.1,
                 percentages_axial: list[float] = [95, 90, 85, 75, 60, 50, 40, 30, 20, 15, 10, 5],
                 percentages_circ: list[float] = [90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 15, 10],
                 percentages_area: list[float] = [85, 75, 60, 50, 40, 30, 20, 15, 10],
                 file_path: str | None = None
                 ):
        """
        Initialize the DentProfiles class.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing dent contour with df.index is 'Axial Displacement', df.columns is 'Circumferential Orientation', and df.values is 'Radius'.
        OD : float
            Outer Diameter of the pipe (inches).
        WT : float
            Wall Thickness of the pipe (inches).
        equal_baseline : bool, optional (default=False)
            If True, use the same baseline for US and DS segments.
        ignore_edge : float, optional (default=0.1)
            Fraction of data to ignore at each edge when determining nominal radius.
        percentages_axial : list of float, optional
            Percentage levels for axial measurements.
        percentages_circ : list of float, optional
            Percentage levels for circumferential measurements.
        percentages_area : list of float, optional
            Percentage levels for area measurements.
        file_path : str or None, optional
            If provided, path to save the generated figures.
        """
        self._df = df
        self._OD = OD
        self._WT = WT
        self._expected_nominal = OD/2 - WT
        self._equal_baseline = equal_baseline
        self._ignore_edge = ignore_edge

        self._percentages_axial = percentages_axial
        self._percentages_circ = percentages_circ
        self._percentages_area = percentages_area
        self._file_path = file_path

        self._prepare_data()
        self._measure_data()
        self._calculate_results()

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
            f"-----------\n"
            f"Summary of Lengths and Areas by Quadrant (% levels as index):\n"
            f"{summary_df.to_string()}\n"
            # Insert the Restraint Parameters here
            f"-----------\n"
            f"Restraint Parameters:\n"
            f" - US_CCW: {round(self._rp['US_CCW'], 3):>8.3f}\n"
            f" - US_CW:  {round(self._rp['US_CW'], 3):>8.3f}\n"
            f" - DS_CCW: {round(self._rp['DS_CCW'], 3):>8.3f}\n"
            f" - DS_CW:  {round(self._rp['DS_CW'], 3):>8.3f}\n"
            f"\n"
        )
        return return_string

    def _prepare_data(self):
        # Locate the deepest point, using the ignore_edge parameter to avoid edge effects
        start_idx = math.ceil(self._df.shape[0]*self._ignore_edge)
        end_idx = math.floor(self._df.shape[0]*(1-self._ignore_edge))
        df_trim = self._df.iloc[start_idx:end_idx, :]
        min_idx = df_trim.stack().idxmin()
        if isinstance(min_idx, tuple):
            self._axial_min, self._circ_min = float(min_idx[0]), float(min_idx[1])
        else:
            # If idxmin returns a single value, find the 2D location manually
            stacked = df_trim.stack()
            self._axial_min = float(stacked.loc[min_idx:min_idx].index[0][0])
            self._circ_min = float(stacked.loc[min_idx:min_idx].index[0][1])
        self._radius_min = float(df_trim.at[self._axial_min, self._circ_min]) # type: ignore
        # Extract the Axial and Circumferential profiles at the deepest point
        self._axial_profile = self._df[self._circ_min]
        self._circ_profile = self._df.loc[self._axial_min]
        # Split Axial data into US/DS
        self._axial_us = self._axial_profile.loc[:self._axial_min]
        self._axial_ds = self._axial_profile.loc[self._axial_min:]
        # Split Circumferential data into CCW/CW
        self._circ_ccw = pd.Series(self._circ_profile.loc[:self._circ_min]) # type: ignore
        self._circ_cw = pd.Series(self._circ_profile.loc[self._circ_min:]) # type: ignore
        # Determine the nominal internal radius
        self._nominal_radius = self.get_nominal(expected_nominal=self._expected_nominal, ignore_edge=self._ignore_edge)
        self._dent_depth = self._nominal_radius - self._radius_min
        # Ensure that dent depth is non-negative
        if self._dent_depth < 0:
            raise ValueError("Calculated dent depth is negative. Check the nominal radius and data for correctness.")
        self._dent_depth_percent = (self._dent_depth / self._OD) * 100

    def _measure_data(self):
        # Determine the baseline index and radii for all four quadrants (index, radius)
        self._baseline_us = self.get_baseline(self._axial_us, axial_circ="axial")
        self._baseline_ds = self.get_baseline(self._axial_ds, axial_circ="axial")
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
        # If equal_baseline is True, find the DS baseline matching the US baseline radius
        if self._equal_baseline:
            confirmation = self.set_baseline_by_radius("DS", self._baseline_us[2], search_direction="outward")
            if not confirmation:
                # Attempt to make US match DS instead
                confirmation = self.set_baseline_by_radius("US", self._baseline_ds[2], search_direction="outward")
                if not confirmation:
                    raise ValueError("Unable to set equal baselines between US and DS segments.")
        # Iterate through all four quadrants to determine lengths and areas
        self._results_axial_us = self.get_measurements(self._axial_us, self._dent_depth_us, self._axial_min, self._baseline_us, self._percentages_axial, self._percentages_area)
        self._results_axial_ds = self.get_measurements(self._axial_ds, self._dent_depth_ds, self._axial_min, self._baseline_ds, self._percentages_axial, self._percentages_area, outbound_data=True)
        self._results_circ_us_ccw = self.get_measurements(self._circ_ccw, self._dent_depth_us_ccw, self._circ_min, self._baseline_us_ccw, self._percentages_circ, self._percentages_area)
        self._results_circ_us_cw = self.get_measurements(self._circ_cw, self._dent_depth_us_cw, self._circ_min, self._baseline_us_cw, self._percentages_circ, self._percentages_area, outbound_data=True)
        self._results_circ_ds_ccw = self.get_measurements(self._circ_ccw, self._dent_depth_ds_ccw, self._circ_min, self._baseline_ds_ccw, self._percentages_circ, self._percentages_area)
        self._results_circ_ds_cw = self.get_measurements(self._circ_cw, self._dent_depth_ds_cw, self._circ_min, self._baseline_ds_cw, self._percentages_circ, self._percentages_area, outbound_data=True)
        # Create three figures
        if self._file_path is not None:
            self.create_lengths_figure("Axial", self._axial_us, self._axial_ds, self._results_axial_us, self._results_axial_ds, self._axial_min, self._file_path)
            self.create_lengths_figure("Circ_US", self._circ_ccw, self._circ_cw, self._results_circ_us_ccw, self._results_circ_us_cw, self._circ_min, self._file_path)
            self.create_lengths_figure("Circ_DS", self._circ_ccw, self._circ_cw, self._results_circ_ds_ccw, self._results_circ_ds_cw, self._circ_min, self._file_path)
    
    def _calculate_results(self):
        self._rp = {
            "US_CCW": get_restraint_parameter(
                AAX_15=self._results_axial_us["areas"][15],
                ATR_15=self._results_circ_us_ccw["areas"][15],
                LTR_70=self._results_circ_us_ccw["lengths"][70]["length"],
                LAX_15=self._results_axial_us["lengths"][15]["length"],
                LAX_30=self._results_axial_us["lengths"][30]["length"],
                LAX_50=self._results_axial_us["lengths"][50]["length"],
                LTR_80=self._results_circ_us_ccw["lengths"][80]["length"]
            ),
            "US_CW": get_restraint_parameter(
                AAX_15=self._results_axial_us["areas"][15],
                ATR_15=self._results_circ_us_cw["areas"][15],
                LTR_70=self._results_circ_us_cw["lengths"][70]["length"],
                LAX_15=self._results_axial_us["lengths"][15]["length"],
                LAX_30=self._results_axial_us["lengths"][30]["length"],
                LAX_50=self._results_axial_us["lengths"][50]["length"],
                LTR_80=self._results_circ_us_cw["lengths"][80]["length"],
            ),
            "DS_CCW": get_restraint_parameter(
                AAX_15=self._results_axial_ds["areas"][15],
                ATR_15=self._results_circ_ds_ccw["areas"][15],
                LTR_70=self._results_circ_ds_ccw["lengths"][70]["length"],
                LAX_15=self._results_axial_ds["lengths"][15]["length"],
                LAX_30=self._results_axial_ds["lengths"][30]["length"],
                LAX_50=self._results_axial_ds["lengths"][50]["length"],
                LTR_80=self._results_circ_ds_ccw["lengths"][80]["length"]
            ),
            "DS_CW": get_restraint_parameter(
                AAX_15=self._results_axial_ds["areas"][15],
                ATR_15=self._results_circ_ds_cw["areas"][15],
                LTR_70=self._results_circ_ds_cw["lengths"][70]["length"],
                LAX_15=self._results_axial_ds["lengths"][15]["length"],
                LAX_30=self._results_axial_ds["lengths"][30]["length"],
                LAX_50=self._results_axial_ds["lengths"][50]["length"],
                LTR_80=self._results_circ_ds_cw["lengths"][80]["length"],
            ),
        }
    
    def graph_lengths(self, quadrant: str):
        """Generate and display a matplotlib Figure for the specified quadrant ('Axial', 'Circ_US', 'Circ_DS')."""
        if quadrant == "Axial":
            self.create_lengths_figure("Axial", self._axial_us, self._axial_ds, self._results_axial_us, self._results_axial_ds, self._axial_min)
        elif quadrant == "Circ_US":
            self.create_lengths_figure("Circ_US", self._circ_ccw, self._circ_cw, self._results_circ_us_ccw, self._results_circ_us_cw, self._circ_min)
        elif quadrant == "Circ_DS":
            self.create_lengths_figure("Circ_DS", self._circ_ccw, self._circ_cw, self._results_circ_ds_ccw, self._results_circ_ds_cw, self._circ_min)
        else:
            raise ValueError("Invalid quadrant specified. Choose from 'Axial', 'Circ_US', 'Circ_DS'.")
    
    def get_profile_data(self, quadrant: str) -> tuple[pd.Series, tuple[int, float, float]] | None:
        """
        Get the profile data and baseline information for a specific quadrant.
        Useful for GUI applications that need to display and interact with profile data.
        
        Parameters
        ----------
        quadrant : str
            The quadrant to retrieve. Options: "US", "DS", "US_CCW", "US_CW", "DS_CCW", "DS_CW"
            
        Returns
        -------
        tuple or None
            Tuple of (profile_data: pd.Series, baseline: tuple[index, position, radius])
            Returns None if invalid quadrant specified.
        """
        data_map = {
            "US": (self._axial_us, self._baseline_us),
            "DS": (self._axial_ds, self._baseline_ds),
            "US_CCW": (self._circ_ccw, self._baseline_us_ccw),
            "US_CW": (self._circ_cw, self._baseline_us_cw),
            "DS_CCW": (self._circ_ccw, self._baseline_ds_ccw),
            "DS_CW": (self._circ_cw, self._baseline_ds_cw),
        }
        
        quadrant_upper = quadrant.upper()
        if quadrant_upper not in data_map:
            return None
            
        return data_map[quadrant_upper]
    
    def set_baseline_by_index(self, US_or_DS: str, index: int) -> bool:
        """
        Manually set the baseline for either US or DS using an index position.
        
        Parameters
        ----------
        US_or_DS : str
            The segment to update. Options: "US", "DS"
        index : int
            The index in the profile data to use as the new baseline.
            
        Returns
        -------
        bool
            True if baseline was successfully updated, False otherwise.
        """
        try:
            if US_or_DS.upper() == "US":
                if index < 0 or index >= len(self._axial_us):
                    return False
                axial_pos = float(self._axial_us.index[index])
                radius = float(self._axial_us.iloc[index])
                self._baseline_us = (index, axial_pos, radius)
                self._dent_depth_us = radius - self._radius_min
                # Update circumferential baselines that depend on US
                self._baseline_us_ccw = self.get_baseline_circ(self._circ_ccw, radius)
                self._baseline_us_cw = self.get_baseline_circ(self._circ_cw, radius, outbound_data=True)
                self._dent_depth_us_ccw = self._baseline_us_ccw[2] - self._radius_min
                self._dent_depth_us_cw = self._baseline_us_cw[2] - self._radius_min
                
            elif US_or_DS.upper() == "DS":
                if index < 0 or index >= len(self._axial_ds):
                    return False
                axial_pos = float(self._axial_ds.index[index])
                radius = float(self._axial_ds.iloc[index])
                self._baseline_ds = (index, axial_pos, radius)
                self._dent_depth_ds = radius - self._radius_min
                # Update circumferential baselines that depend on DS
                self._baseline_ds_ccw = self.get_baseline_circ(self._circ_ccw, radius)
                self._baseline_ds_cw = self.get_baseline_circ(self._circ_cw, radius, outbound_data=True)
                self._dent_depth_ds_ccw = self._baseline_ds_ccw[2] - self._radius_min
                self._dent_depth_ds_cw = self._baseline_ds_cw[2] - self._radius_min
            else:
                return False
                
            return True
        except Exception:
            return False
    
    def set_baseline_by_position(self, US_or_DS: str, position: float) -> bool:
        """
        Manually set the baseline for either US or DS using an axial position.
        The function will find the closest data point to the specified position.
        
        Parameters
        ----------
        US_or_DS : str
            The segment to update. Options: "US", "DS"
        position : float
            The axial (inches) position to use as the new baseline.
            
        Returns
        -------
        bool
            True if baseline was successfully updated, False otherwise.
        """
        try:
            if US_or_DS.upper() == "US":
                closest_idx = int(abs(self._axial_us.index - position).argmin()) # type: ignore
                return self.set_baseline_by_index(US_or_DS, closest_idx)
                
            elif US_or_DS.upper() == "DS":
                closest_idx = int(abs(self._axial_ds.index - position).argmin()) # type: ignore
                return self.set_baseline_by_index(US_or_DS, closest_idx)
            
            else:
                return False
        except Exception:
            return False
    
    def set_baseline_by_radius(self, US_or_DS: str, radius: float, search_direction: str = "outward") -> bool:
        """
        Manually set the baseline for either US or DS by finding the nearest point with the specified radius.
        
        Parameters
        ----------
        US_or_DS : str
            The segment to update. Options: "US", "DS"
        radius : float
            The target radius value (inches) to find in the profile.
        search_direction : str, optional
            Direction to search from dent minimum. "outward" (default) searches away from dent,
            "inward" searches toward dent.
            
        Returns
        -------
        bool
            True if baseline was successfully updated, False otherwise.
        """
        try:
            if US_or_DS.upper() == "US":
                data = self._axial_us if search_direction == "outward" else self._axial_us.iloc[::-1]
                closest_idx = int((data - radius).abs().argmin())
                # Convert back to original indexing if reversed
                if search_direction != "outward":
                    closest_idx = len(self._axial_us) - 1 - closest_idx
                return self.set_baseline_by_index(US_or_DS, closest_idx)
                
            elif US_or_DS.upper() == "DS":
                data = self._axial_ds if search_direction == "outward" else self._axial_ds.iloc[::-1]
                closest_idx = int((data - radius).abs().argmin())
                if search_direction != "outward":
                    closest_idx = len(self._axial_ds) - 1 - closest_idx
                return self.set_baseline_by_index(US_or_DS, closest_idx)
                
            else:
                return False
        except Exception:
            return False
        
    def reset_baselines(self):
        """
        Reset the baselines for all quadrants to their original calculated values.
        """
        self._prepare_data()
        self._measure_data()
        self._calculate_results()
    
    def recalculate_measurements(self, US_DS: list[str] | None = None):
        """
        Recalculate all measurements (lengths and areas) after baseline changes.
        
        Parameters
        ----------
        US_DS : list of str, optional
            List of segments to recalculate. If None, recalculates all segments.
            Options: ["US", "DS"]
        """
        if US_DS is None:
            US_DS = ["US", "DS"]
        
        for segment in US_DS:
            if segment.upper() == "US":
                # US Axial
                self._results_axial_us = self.get_measurements(
                    self._axial_us, self._dent_depth_us, self._axial_min, 
                    self._baseline_us, self._percentages_axial, self._percentages_area
                )
                # US CCW
                self._results_circ_us_ccw = self.get_measurements(
                    self._circ_ccw, self._dent_depth_us_ccw, self._circ_min, 
                    self._baseline_us_ccw, self._percentages_circ, self._percentages_area
                )
                # US CW
                self._results_circ_us_cw = self.get_measurements(
                    self._circ_cw, self._dent_depth_us_cw, self._circ_min, 
                    self._baseline_us_cw, self._percentages_circ, self._percentages_area, 
                    outbound_data=True
                )
            elif segment.upper() == "DS":
                # DS Axial
                self._results_axial_ds = self.get_measurements(
                    self._axial_ds, self._dent_depth_ds, self._axial_min, 
                    self._baseline_ds, self._percentages_axial, self._percentages_area, 
                    outbound_data=True
                )
                # DS CCW
                self._results_circ_ds_ccw = self.get_measurements(
                    self._circ_ccw, self._dent_depth_ds_ccw, self._circ_min, 
                    self._baseline_ds_ccw, self._percentages_circ, self._percentages_area
                )
                # DS CW
                self._results_circ_ds_cw = self.get_measurements(
                    self._circ_cw, self._dent_depth_ds_cw, self._circ_min, 
                    self._baseline_ds_cw, self._percentages_circ, self._percentages_area, 
                    outbound_data=True
                )
            else:
                raise ValueError("Invalid segment specified for recalculation. Choose from 'US', 'DS'.")
        
        self._calculate_results()
    
    def get_baseline_info(self, quadrant: str) -> dict | None:
        """
        Get current baseline information for a specific quadrant.
        
        Parameters
        ----------
        quadrant : str
            The quadrant to query. Options: "US", "DS", "US_CCW", "US_CW", "DS_CCW", "DS_CW"
            
        Returns
        -------
        dict or None
            Dictionary with keys: 'index', 'position', 'radius', 'dent_depth'
            Returns None if invalid quadrant specified.
        """
        baseline_map = {
            "US": (self._baseline_us, self._dent_depth_us),
            "DS": (self._baseline_ds, self._dent_depth_ds),
            "US_CCW": (self._baseline_us_ccw, self._dent_depth_us_ccw),
            "US_CW": (self._baseline_us_cw, self._dent_depth_us_cw),
            "DS_CCW": (self._baseline_ds_ccw, self._dent_depth_ds_ccw),
            "DS_CW": (self._baseline_ds_cw, self._dent_depth_ds_cw),
        }
        
        quadrant_upper = quadrant.upper()
        if quadrant_upper not in baseline_map:
            return None
            
        baseline, dent_depth = baseline_map[quadrant_upper]
        return {
            'index': baseline[0],
            'position': baseline[1],
            'radius': baseline[2],
            'dent_depth': dent_depth
        }
    
    def validate_baseline(self, US_or_DS: str, index: int) -> dict:
        """
        Validate if a proposed baseline index is reasonable without applying it.
        
        Parameters
        ----------
        US_or_DS : str
            The segment to validate. Options: "US", "DS"
        index : int
            The proposed index to validate.
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'valid': bool indicating if the baseline is valid
            - 'reason': str with explanation if invalid
            - 'radius': float with radius at that index
            - 'dent_depth': float with resulting dent depth
        """
        result = {'valid': False, 'reason': '', 'radius': None, 'dent_depth': None}
        
        try:
            # Get the appropriate data series
            data_map = {
                "US": self._axial_us,
                "DS": self._axial_ds,
            }
            
            US_or_DS_upper = US_or_DS.upper()
            if US_or_DS_upper not in data_map:
                result['reason'] = 'Invalid segment specified'
                return result
                
            data = data_map[US_or_DS_upper]
            
            # Check index bounds
            if index < 0 or index >= len(data):
                result['reason'] = f'Index {index} out of bounds [0, {len(data)-1}]'
                return result
            
            radius = float(data.iloc[index])
            result['radius'] = radius
            
            # Calculate resulting dent depth
            dent_depth = radius - self._radius_min
            result['dent_depth'] = dent_depth
            
            # Validate that radius is greater than minimum
            if radius <= self._radius_min:
                result['reason'] = f'Baseline radius ({radius:.4f}) must be greater than dent minimum ({self._radius_min:.4f})'
                return result
            
            # Validate that radius is not too far from nominal (e.g., within 20% of nominal)
            if radius > self._nominal_radius * 1.2:
                result['reason'] = f'Baseline radius ({radius:.4f}) is unreasonably high (>120% of nominal: {self._nominal_radius:.4f})'
                return result
            
            # Validate that dent depth is reasonable (positive)
            if dent_depth <= 0:
                result['reason'] = f'Resulting dent depth ({dent_depth:.4f}) must be positive'
                return result
            
            result['valid'] = True
            result['reason'] = 'Baseline is valid'
            return result
            
        except Exception as e:
            result['reason'] = f'Error during validation: {str(e)}'
            return result

    def get_nominal(self, expected_nominal: float, threshold: float = 0.01, ignore_edge: float = 0.1) -> float:
        """
        Determine the nominal radius from the profile data.

        Parameters
        ----------
        expected_nominal : float
            Expected nominal radius to validate against.
        threshold : float, optional (default=0.01)
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
        mean_value = edge_data.stack().mean()
        nominal_radius = float(mean_value.item() if isinstance(mean_value, pd.Series) else mean_value)
        if expected_nominal is not None and abs(nominal_radius - expected_nominal) <= (expected_nominal * threshold):
            # If the expected nominal is provided and close enough, use it
            nominal_radius = expected_nominal
        return nominal_radius

    def get_baseline(self, 
                     data: pd.Series, 
                     axial_circ: str = "axial", 
                     axial_default: float = 0.025, 
                     circ_default: float = -0.15,
                     **kwargs) -> tuple[int, float, float]:
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
        baseline_index, baseline_axial, baseline_val = find_deflection(data, **kwargs)

        if baseline_index is not None and baseline_axial is not None and baseline_val is not None:
            return baseline_index, baseline_axial, baseline_val
        else:
            # If no valid baseline found, return default and find the closest point in data to the default,
            closest_idx = int((data - baseline_default).abs().idxmin())
            if closest_idx is None:
                raise ValueError("Unable to determine baseline index from data.")
            closest_axial = float(data.index[closest_idx])
            return closest_idx, closest_axial, baseline_default

    def get_baseline_circ(self, 
                          data: pd.Series, 
                          baseline_axial_radius: float, 
                          outbound_data: bool = False) -> tuple[int, float, float]:
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
            circ_deg = float(data.index[circ_index])
            circ_radius = float(data.loc[circ_deg])
        else:
            # Linear interpolation to find the exact crossing point
            circ_index = data.index.get_loc(crossing_idx)
            if isinstance(circ_index, slice):
                circ_index = int(circ_index.start)  # Take the start if slice
            elif isinstance(circ_index, np.ndarray):
                # If mask, take the first True occurrence
                circ_index = int(np.where(circ_index)[0][0])
            x0, x1 = float(data.index[circ_index - 1]), float(pd.Series([crossing_idx]).item())
            y0, y1 = float(data.loc[x0]), float(data.loc[x1])
            if y1 != y0:
                circ_deg = x0 + (baseline_axial_radius - y0) * (x1 - x0) / (y1 - y0)
            else:
                circ_deg = x1  # If y1 == y0, just take the crossing index
            circ_radius = baseline_axial_radius

        return circ_index, circ_deg, circ_radius
    
    def get_measurements(self, 
                         data: pd.Series, 
                         dent_depth: float, 
                         dent_location: float, 
                         baseline: tuple, 
                         percentages_length: list[float], 
                         percentages_area: list[float], 
                         outbound_data: bool = False) -> dict:
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
                circ_index = data.index.get_loc(crossing_idx)
                if isinstance(circ_index, slice):
                    circ_index = int(circ_index.start)  # Take the start if slice
                elif isinstance(circ_index, np.ndarray):
                    # If mask, take the first True occurrence
                    circ_index = int(np.where(circ_index)[0][0])
                x0, x1 = float(data.index[circ_index - 1]), float(pd.Series([crossing_idx]).item())
                y0, y1 = float(data.loc[x0]), float(data.loc[x1])
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

    def create_lengths_figure(self, 
                      quadrant: str, 
                      profile_us: pd.Series, 
                      profile_ds: pd.Series, 
                      results_us: dict, 
                      results_ds: dict, 
                      dent_location: float, 
                      file_path: str | None = None, 
                      palette: list = [
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
        ax.plot(profile_us.index, profile_us, label=us_label, color="#000000")
        ax.plot(profile_ds.index, profile_ds, label=ds_label, color="#ff0000", linestyle='--')

        # Plot the nominal radius line
        # ax.axhline(y=self._nominal_radius, color='gray', linestyle=':', linewidth=1, label='Nominal Radius')

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
        
    def graph_contours(self, file_path: str | None = None, palette: str = 'viridis'):
        """
        Create a matplotlib figure showing the dent contour with the minimum point highlighted.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        c = ax.contourf(self._df.index, self._df.columns, self._df.values.T, cmap=palette)
        fig.colorbar(c, ax=ax, label='Radius (in)')
        ax.plot(self._axial_min, self._circ_min, 'ro', label='Dent Minimum')
        ax.set_title('Dent Contour')
        ax.set_ylabel('Circumferential Position (deg)')
        ax.set_xlabel('Axial Position (in)')
        ax.legend()
        fig.tight_layout()
        if file_path:
            fig.savefig(str(file_path).replace('.xlsx', '_Dent_Contour.png'), dpi=300)
            plt.close(fig)

    def rp(self, quadrant: str | None = None) -> float | dict:
        """Calculate the Restraint Parameter (RP) for the specified quadrant or all quadrants if None."""
        quadrant_options = ["US_CCW", "US_CW", "DS_CCW", "DS_CW"]
        quadrant_upper = quadrant.upper() if quadrant is not None else None

        if quadrant_upper is None:
            return self._rp
        elif quadrant_upper in self._rp:
            return self._rp[quadrant_upper]
        else:
            raise ValueError(f"Quadrant '{quadrant}' not found in restraint parameters. Choose from {quadrant_options} or None for all quadrants.")
    @property
    def min_idx(self) -> tuple[int, int]:
        """Tuple of (Axial index, Circumferential index) of the deepest point."""
        axial_idx = self._df.index.get_loc(self._axial_min)
        if isinstance(axial_idx, slice):
            axial_idx = int(axial_idx.start)  # Take the start if slice
        elif isinstance(axial_idx, np.ndarray):
            # If mask, take the first True occurrence
            axial_idx = int(np.where(axial_idx)[0][0])
        circ_idx = self._df.columns.get_loc(self._circ_min)
        if isinstance(circ_idx, slice):
            circ_idx = int(circ_idx.start)  # Take the start if slice
        elif isinstance(circ_idx, np.ndarray):
            # If mask, take the first True occurrence
            circ_idx = int(np.where(circ_idx)[0][0])
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
        return pd.Series(self._circ_profile)
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
    def baseline_us(self) -> tuple[int, float, float]:
        """Baseline radius upstream of the deepest point."""
        return self._baseline_us
    @property
    def baseline_ds(self) -> tuple[int, float, float]:
        """Baseline radius downstream of the deepest point."""
        return self._baseline_ds
    @property
    def baseline_us_ccw(self) -> tuple[int, float, float]:
        """Baseline radius counter-clockwise of the deepest point."""
        return self._baseline_us_ccw
    @property
    def baseline_us_cw(self) -> tuple[int, float, float]:
        """Baseline radius clockwise of the deepest point."""
        return self._baseline_us_cw
    @property
    def baseline_ds_ccw(self) -> tuple[int, float, float]:
        """Baseline radius counter-clockwise of the deepest point."""
        return self._baseline_ds_ccw
    @property
    def baseline_ds_cw(self) -> tuple[int, float, float]:
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
