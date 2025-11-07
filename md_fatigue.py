# coding: utf-8
"""
Implements the fatigue life assessment of dents with and without interacting features
Ref. Papers: 
- PR-214-223806-R01 (MD-5-3)
- PR-214-223810-R01 (MD-2-4)
- PR-214-114500-R01 (MD-4-9)
"""

import math
import json
import os
import numpy as np
import pandas as pd

import cycle_counting as cc
import md_profiles as mpf
import traceback

tables_folder = r"data"

table_md = 'Tables_Coefficients.json'
table_fatigue = 'Tables_FatigueCurves.json'

restraint_types = ["Unrestrained", "Shallow Restrained", "Deep Restrained", "Restrained", "Mixed"]

def _get_km_l0(OD: float, WT: float, restraint: str, max_pct_psmys: float) -> float:
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")

    if max_pct_psmys <= 1:
        max_pct_psmys *= 100

    r = OD / WT

    restraint = restraint.replace("-", " ").replace(chr(8211), " ").strip().lower()

    if restraint == "unrestrained":
        if max_pct_psmys <= 20:
            km = 9.4 * (1 - math.exp(-0.045 * r))  # (A1)
        else:
            km = 7.5 * (1 - math.exp(-0.065 * r))  # (A2)
    elif restraint == "shallow restrained":
        km = 0.1183 * r - 1.146  # (A3)
    elif restraint == "deep restrained":
        km = 0.1071 * r + 0.1332  # (A4)
    elif restraint == "restrained":
        km_sh = 0.1183 * r - 1.146
        km_dp = 0.1071 * r + 0.1332
        km = max(km_sh, km_dp)
    elif "mixed" in restraint:
        if max_pct_psmys <= 20:
            result_unres = 9.4 * (1 - math.exp(-0.045 * r))  # (A1)
        else:
            result_unres = 7.5 * (1 - math.exp(-0.065 * r))  # (A2)
        if "shallow" in restraint:
            result_res = 0.1183 * r - 1.146  # (A3)
        elif "deep" in restraint:
            result_res = 0.1071 * r + 0.1332  # (A4)
        else:
            raise ValueError("Mixed restraint type must specify shallow or deep")
        km = max(result_unres, result_res)
    else:
        raise ValueError("Unknown restraint type")

    return km

def _get_a5(OD: float, WT: float, dP: float, tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Level 0.5 Unrestrained
    Coefficients from Table A.2
    """
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")

    # Load coefficients from JSON
    with open(tables_path, 'r') as f:
        data = json.load(f)

    # Assuming the structure: data["A.2"][0] contains the dict with keys like "a00", "a10", etc.
    coeffs = data["A.2"][0]

    x = dP / 100
    y = OD / WT

    result = (
        coeffs["a00"] +
        coeffs["a10"] * x +
        coeffs["a01"] * y +
        coeffs["a20"] * x ** 2 +
        coeffs["a11"] * x * y +
        coeffs["a02"] * y ** 2 +
        coeffs["a30"] * x ** 3 +
        coeffs["a21"] * x ** 2 * y +
        coeffs["a12"] * x * y ** 2
    )

    return result

def _get_a6(OD: float, WT: float, dP: float, tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Level 0.5 Shallow Restrained
    Coefficients from Table A.3
    """
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    
    # Load coefficients from JSON
    with open(tables_path, 'r') as f:
        data = json.load(f)

    # Assuming the structure: data["A.3"][0] contains the dict with keys like "a0", "a1", etc.
    coeffs = data["A.3"][0]

    x = dP / 100
    y = OD / WT

    poly_part = (
        coeffs["a0"] +
        coeffs["a1"] * x +
        coeffs["a2"] * y +
        coeffs["a3"] * x * y +
        coeffs["a4"] * x ** 2 +
        coeffs["a5"] * y ** 2 +
        coeffs["a6"] * x ** 3 +
        coeffs["a7"] * x ** 2 * y +
        coeffs["a8"] * x * y ** 2 +
        coeffs["a9"] * y ** 3
    )

    exp_part = (
        math.exp(-abs(coeffs["a10"] * x)) +
        math.exp(-abs(coeffs["a11"] * y + coeffs["a12"] * y ** 2 + coeffs["a13"] * y ** 3))
    )

    result = poly_part * exp_part

    return result

def _get_a7(OD: float, WT: float, dP: float, tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Level 0.5 Deep Restrained
    Coefficients from Table A.4
    """
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    
    # Load coefficients from JSON
    with open(tables_path, 'r') as f:
        data = json.load(f)
    
    # Assuming the structure: data["A.4"][0] contains the dict with keys like "a1", "a2", etc.
    coeffs = data["A.4"][0]

    x = dP / 100
    y = OD / WT

    result = (
        coeffs["a1"] +
        coeffs["a2"] * x +
        coeffs["a3"] * y +
        coeffs["a4"] * x * y +
        coeffs["a5"] * x ** 2 +
        coeffs["a6"] * y ** 2 +
        coeffs["a7"] * x ** 2 * y +
        coeffs["a8"] * x * y ** 2 +
        coeffs["a9"] * x ** 3 +
        coeffs["a10"] * y ** 3
    )

    return result

def _get_km_l05(OD: float, WT: float, dP: float, restraint: str) -> float:
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")

    restraint = restraint.strip().lower()

    if restraint == "unrestrained":
        return _get_a5(OD, WT, dP)
    elif restraint == "shallow restrained":
        return _get_a6(OD, WT, dP)
    elif restraint == "deep restrained":
        return _get_a7(OD, WT, dP)
    elif "mixed" in restraint:
        result_unres = _get_a5(OD, WT, dP)
        result_res = 0
        if "shallow" in restraint:
            result_res = _get_a6(OD, WT, dP)
        elif "deep" in restraint:
            result_res = _get_a7(OD, WT, dP)
        return max(result_unres, result_res)
    else:
        raise ValueError("Unknown restraint condition")

def _get_a8(OD: float, WT: float, dP: float, Pmean: float, restraint: str, tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Level 0.5+
    Coefficients from Table A.5
    """
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")
    
    # Load coefficients from JSON
    with open(tables_path, 'r') as f:
        data = json.load(f)

    table = data["A.5"]

    x = dP / 100
    y = Pmean / 100
    z = OD / WT
    odt = OD / WT

    # Normalize restraint for matching
    restraint_condition = restraint.strip().title()

    # Find the matching row
    coeffs = None
    for row in table:
        if row["Restraint"].strip().title() == restraint_condition and row["ODot_LB"] <= odt < row["ODot_UB"]:
            coeffs = row
            break

    if not coeffs:
        raise ValueError("Out of range, no matching coefficients found")

    # Extract a1 to a14
    a = [coeffs[f"a{i}"] for i in range(1, 15)]

    # Calculate parts
    part1 = a[0] + a[1] * x + a[2] * y + a[3] * x * y + a[4] * x ** 2 + a[5] * y ** 2
    part2 = a[6] + a[7] * z + a[8] * z ** 2
    part3 = a[9] + a[10] * x + a[11] * y + a[12] * x ** 2 + a[13] * y ** 2

    result = part1 * part2 * math.exp(-abs(part3))

    return result

def get_km_l05p(OD: float, WT: float, dP: float, Pmean: float, restraint: str) -> float:
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")

    restraint = restraint.strip().lower()

    if restraint in ["unrestrained", "shallow restrained", "deep restrained"]:
        return _get_a8(OD, WT, dP, Pmean, restraint.title())
    elif "mixed" in restraint:
        result_unres = _get_a8(OD, WT, dP, Pmean, "Unrestrained")
        result_res = 0
        if "shallow" in restraint:
            result_res = _get_a8(OD, WT, dP, Pmean, "Shallow Restrained")
        elif "deep" in restraint:
            result_res = _get_a8(OD, WT, dP, Pmean, "Deep Restrained")
        return max(result_unres, result_res)
    else:
        raise ValueError("Unknown restraint condition")

def _get_a9(OD: float, WT: float, dP: float, depth: float, restraint: str, tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Level 0.75
    Coefficients from Table A.6
    """
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")
    
    # Load coefficients from JSON
    with open(tables_path, 'r') as f:
        data = json.load(f)

    table = data["A.6"]

    x = dP / 100
    y = OD / (WT * 100)
    odt = OD / WT
    d = depth

    # Determine which row to use based on restraint condition and OD/t ratio
    # Restraint Condition can be either "Restrained" or "Unrestrained"
    restraint_condition = "Unrestrained" if "unrestrained" in restraint.lower() else "Restrained"
    restraint_condition = restraint_condition.strip().title()
    
    # Find the matching row
    coeffs = None
    for row in table:
        if (row["Restraint"].strip().title() == restraint_condition and 
            row["ODot_LB"] <= odt < row["ODot_UB"]):
            coeffs = row
            break

    if not coeffs:
        raise ValueError("Out of range, no matching coefficients found")

    # Extract b1 to b15 coefficients
    b = [coeffs[f"b{i}"] for i in range(1, 16)]

    # Calculate intermediate values c1 through c5
    c1 = b[0] + (b[1] * x) + (b[2] * y)
    c2 = b[3] + (b[4] * x) + (b[5] * y)
    c3 = b[6] + (b[7] * x) + (b[8] * y)
    c4 = b[9] + (b[10] * x) + (b[11] * y)
    c5 = b[12] + (b[13] * x) + (b[14] * y)

    # Calculate final result
    result = abs(c1) - abs(c2) * ((abs(c3) - (abs(c4) * d) ** 2) * math.exp(-abs(c5) * d))

    return result

def _get_km_l075(OD: float, WT: float, dP: float, depth: float, restraint: str) -> float:
    """
    Level 0.75
    Return the max Km value
    """
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")

    restraint = restraint.strip().lower()

    if restraint in ["unrestrained", "shallow restrained", "deep restrained"]:
        return _get_a9(OD, WT, dP, depth, restraint.title())
    elif "mixed" in restraint:
        result_unres = _get_a9(OD, WT, dP, depth, "Unrestrained")
        result_res = 0
        
        if "shallow" in restraint:
            result_res = _get_a9(OD, WT, dP, depth, "Shallow Restrained")
        elif "deep" in restraint:
            result_res = _get_a9(OD, WT, dP, depth, "Deep Restrained")
            
        return max(result_unres, result_res)
    else:
        raise ValueError("Unknown restraint condition")
    
def _get_a10(OD: float, WT: float, dP: float, pmean: float, depth: float, restraint: str, tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Level 0.75+
    Coefficients from Table A.7
    """
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")
    
    # Load coefficients from JSON
    with open(tables_path, 'r') as f:
        data = json.load(f)

    table = data["A.7"]

    x = dP / 100
    y = pmean / 100
    z = OD / (WT * 100)
    odt = OD / WT
    d = depth

    # Determine which row to use based on restraint condition and OD/t ratio
    # Restraint Condition can be either "Restrained" or "Unrestrained"
    restraint_condition = "Unrestrained" if "unrestrained" in restraint.lower() else "Restrained"
    restraint_condition = restraint_condition.strip().title()
    
    # Find the matching row
    coeffs = None
    for row in table:
        if (row["Restraint"].strip().title() == restraint_condition and 
            row["ODot_LB"] <= odt < row["ODot_UB"]):
            coeffs = row
            break

    if not coeffs:
        raise ValueError("Out of range, no matching coefficients found")

    # Extract b1 to b20 coefficients
    b = [coeffs[f"b{i}"] for i in range(1, 21)]

    # Calculate intermediate values c1 through c5
    c1 = b[0] + b[1] * x + b[2] * y + b[3] * z
    c2 = b[4] + b[5] * x + b[6] * y + b[7] * z
    c3 = b[8] + b[9] * x + b[10] * y + b[11] * z
    c4 = b[12] + b[13] * x + b[14] * y + b[15] * z
    c5 = b[16] + b[17] * x + b[18] * y + b[19] * z

    # Calculate final result
    result = abs(c1) - abs(c2) * ((abs(c3) - (abs(c4) * d) ** 2) * math.exp(-abs(c5) * d))

    return result

def _get_km_l075p(OD: float, WT: float, dP: float, pmean: float, depth: float, restraint: str) -> float:
    """
    Level 0.75+ - wrapper function for _get_a10
    Return the max Km value
    """
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")

    restraint = restraint.strip().lower()

    if restraint in ["unrestrained", "shallow restrained", "deep restrained"]:
        return _get_a10(OD, WT, dP, pmean, depth, restraint.title())
    elif "mixed" in restraint:
        result_unres = _get_a10(OD, WT, dP, pmean, depth, "Unrestrained")
        result_res = 0
        
        if "shallow" in restraint:
            result_res = _get_a10(OD, WT, dP, pmean, depth, "Shallow Restrained")
        elif "deep" in restraint:
            result_res = _get_a10(OD, WT, dP, pmean, depth, "Deep Restrained")
            
        return max(result_unres, result_res)
    else:
        raise ValueError("Unknown restraint condition")
    
def _get_a16(OD: float, WT: float, pmean: float, pmax: float, pmin: float, pili: float, smys: float, 
           restraint: str, lax_values: list, ltr_values: list, aax_values: list, atr_values: list,
           tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Level 1 & 2
    Uses coefficients from MD-5-3 Table A.8 and API 1183 Annex G Tables
    """
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")
    
    # Load coefficients from JSON
    with open(tables_path, 'r') as f:
        data = json.load(f)

    # Calculate PF
    pf = (((pmax + pmin) * (pmax - pmin)) / (2 * 10000)) ** (1/3)
    
    # Calculate r
    r = -2.3053 * pf + 1.5685
    
    # Determine M based on restraint condition
    restraint_condition = restraint.strip().title()
    if restraint_condition == "Unrestrained":
        m = 8
    else:
        m = 4
    
    # Calculate GSF
    gsf = (smys / 52) ** m
    
    # Convert ranges to lists (assuming they're already lists in Python)
    lax = lax_values
    ltr = ltr_values
    aax = aax_values
    atr = atr_values
    
    # Calculate xL and xH based on restraint condition
    if restraint_condition == "Deep Restrained":
        xl = ((aax[5] * aax[1]) ** 0.5 / (WT * lax[3])) ** 1.5 * (lax[3] / ltr[3]) ** 0.5
        xh = (aax[8] / (lax[10] * lax[3])) ** 0.75 * (ltr[3] / lax[3])
        
    elif restraint_condition == "Shallow Restrained":
        xl = ((aax[5] * aax[1]) ** 0.5 / (WT * lax[3])) ** 1.5 * (lax[3] / ltr[3]) ** 0.5
        xh = 10 * (ltr[10] / lax[9]) ** 0.5
        
    elif restraint_condition == "Unrestrained":
        # Get lambda values from G tables
        tmp = int(pmean/10) * 10  # Truncate the TMP value to floor
        rilip = round(pili/10) * 10  # Round RILIP value
        
        lambh = lambl = None
        # Search through G.1 to G.7 tables for matching values
        table = data["G.1_7"]
        for row in table:
            if row["TMP"] == tmp and row["RILIP"] == rilip:
                lambh = row["lambH"]
                lambl = row["lambL"]
                break
        
        if lambh is None or lambl is None:
            raise ValueError("Out of range, no matching lambda values found")
        
        xl = 10**4 * lambl * ((aax[0] * aax[1]) / (OD * WT**2 * lax[3])) ** 1.2 * (lax[2] / ltr[1]) ** 1.5
        xh = 10**4 * lambh * ((aax[1] * atr[1]) / (OD * WT * lax[3] * ltr[3])) ** 1.2 * (lax[3] / ltr[3]) ** 1.5
    else:
        raise ValueError("Unknown restraint condition")
    
    # Calculate SP
    if restraint_condition == "Unrestrained":
        sp = (r * xl + (1 - r) * xh) * gsf
    else:
        sp = (r * xl + (1 - r) * xh) * gsf * (OD / WT) ** 0.25
    
    # Get coefficients A and B from Table A.8
    table_a8 = data["A.8"]
    
    a = b = None
    # Search through A.8 table for matching values
    for row in table_a8:
        if (row["Restraint"].strip().title() == restraint_condition and
            row["Pmin_%SMYS"] == pmin and row["Pmax_%SMYS"] == pmax):
            a = row["log10A"]
            b = row["B"]
            break

    if a is None or b is None:
        raise ValueError("Out of range, no matching coefficients found")
    
    # Calculate final result
    result = (10 ** a) * sp ** b
    
    return result
    
def _get_a16_min(restraint_condition: str, od: float, t: float, pmean: float, pmax: float, pmin: float, 
               pili: float, smys: float, us_lax_values: list, us_aax_values: list, ds_lax_values: list, 
               ds_aax_values: list, usccw_ltr_values: list, usccw_atr_values: list, uscw_ltr_values: list, 
               uscw_atr_values: list, dsccw_ltr_values: list, dsccw_atr_values: list, dscw_ltr_values: list, 
               dscw_atr_values: list) -> float:
    """
    Iterate through all four quadrants using the combinations of US-CCW, US-CW, DS-CCW, DS-CW 
    and return the minimum result from _get_a16
    """
    # Error handling for inputs
    if od <= 0 or t <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint_condition.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")
    try:
        # Check if any of the input lists are None or empty
        input_lists = [us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                      usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                      dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values]
        
        if any(lst is None or len(lst) == 0 for lst in input_lists):
            raise ValueError("Invalid input ranges")
        
        # Calculate results for all four quadrants
        results = []
        
        # Quadrant 1: US-CCW
        result1 = _get_a16(od, t, pmean, pmax, pmin, pili, smys, restraint_condition, 
                         us_lax_values, usccw_ltr_values, us_aax_values, usccw_atr_values)
        results.append(result1)
        
        # Quadrant 2: US-CW
        result2 = _get_a16(od, t, pmean, pmax, pmin, pili, smys, restraint_condition, 
                         us_lax_values, uscw_ltr_values, us_aax_values, uscw_atr_values)
        results.append(result2)
        
        # Quadrant 3: DS-CCW
        result3 = _get_a16(od, t, pmean, pmax, pmin, pili, smys, restraint_condition, 
                         ds_lax_values, dsccw_ltr_values, ds_aax_values, dsccw_atr_values)
        results.append(result3)
        
        # Quadrant 4: DS-CW
        result4 = _get_a16(od, t, pmean, pmax, pmin, pili, smys, restraint_condition, 
                         ds_lax_values, dscw_ltr_values, ds_aax_values, dscw_atr_values)
        results.append(result4)
        
        # Filter out error results and find minimum
        valid_results = []
        for result in results:
            if isinstance(result, (int, float)) and not isinstance(result, str):
                valid_results.append(result)
        
        # Check if any valid results were found
        if valid_results:
            return min(valid_results)
        else:
            raise ValueError("No valid results found")
    
    except Exception as e:
        raise ValueError(f"Error in calculation: {str(e)}")
    
def _get_n_l2(restraint_condition: str, od: float, t: float, pmean: float, pmax: float, pmin: float, 
            pili: float, smys: float, us_lax_values: list, us_aax_values: list, ds_lax_values: list, 
            ds_aax_values: list, usccw_ltr_values: list, usccw_atr_values: list, uscw_ltr_values: list, 
            uscw_atr_values: list, dsccw_ltr_values: list, dsccw_atr_values: list, dscw_ltr_values: list, 
            dscw_atr_values: list) -> float:
    """
    Returns the minimum Cycles to Failure for Level 2
    """
    # Error handling for inputs
    if od <= 0 or t <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint_condition.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")
    try:
        # Check for empty or invalid restraint condition
        if not restraint_condition or restraint_condition.strip().lower() == "error":
            raise ValueError("Invalid restraint condition")

        restraint_condition = restraint_condition.strip().lower()

        if restraint_condition in ["unrestrained", "shallow restrained", "deep restrained"]:
            return _get_a16_min(restraint_condition.title(), od, t, pmean, pmax, pmin, pili, smys,
                              us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                              usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                              dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
        
        elif "mixed" in restraint_condition:
            result_unres = _get_a16_min("Unrestrained", od, t, pmean, pmax, pmin, pili, smys,
                                      us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                      usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                      dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            
            result_res = 0
            
            if "shallow" in restraint_condition:
                result_res = _get_a16_min("Shallow Restrained", od, t, pmean, pmax, pmin, pili, smys,
                                        us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                        usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                        dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            elif "deep" in restraint_condition:
                result_res = _get_a16_min("Deep Restrained", od, t, pmean, pmax, pmin, pili, smys,
                                        us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                        usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                        dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            
            # Check if either result is an error string
            if isinstance(result_unres, str) or isinstance(result_res, str):
                raise ValueError("Error in calculation")
                
            return min(result_unres, result_res)
        else:
            raise ValueError("Unknown restraint condition")
            
    except Exception as e:
        raise ValueError(f"Error in calculation: {str(e)}")
    
def _get_a16_min_l1(restraint_condition: str, od: float, t: float, closest_pmean: float, pili: float, 
                      smys: float, us_lax_values: list, us_aax_values: list, ds_lax_values: list, 
                      ds_aax_values: list, usccw_ltr_values: list, usccw_atr_values: list, uscw_ltr_values: list, 
                      uscw_atr_values: list, dsccw_ltr_values: list, dsccw_atr_values: list, dscw_ltr_values: list, 
                      dscw_atr_values: list) -> float:
    """
    Choose the applicable case, then iterate through the four quadrants using _get_a16_min
    Level 1 specific function that maps ClosestPMean to Pmin/Pmax ranges
    """
    # Error handling for inputs
    if od <= 0 or t <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint_condition.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")
    try:
        # Determine Pmin and Pmax based on ClosestPMean
        if closest_pmean == 25:
            pmin = 10
            pmax = 40
        elif closest_pmean == 45:
            pmin = 30
            pmax = 60
        elif closest_pmean == 65:
            pmin = 50
            pmax = 80
        else:
            raise ValueError("Invalid ClosestPMean value")
        
        # Call _get_a16_min with the determined pressure values
        return _get_a16_min(restraint_condition, od, t, closest_pmean, pmax, pmin, pili, smys,
                          us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                          usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                          dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)

    except Exception as e:
        raise ValueError(f"Error in calculation: {str(e)}")
    
def _get_n_l1(restraint_condition: str, od: float, t: float, closest_pmean: float, pili: float, 
            smys: float, us_lax_values: list, us_aax_values: list, ds_lax_values: list, 
            ds_aax_values: list, usccw_ltr_values: list, usccw_atr_values: list, uscw_ltr_values: list, 
            uscw_atr_values: list, dsccw_ltr_values: list, dsccw_atr_values: list, dscw_ltr_values: list, 
            dscw_atr_values: list) -> float:
    """
    Returns the minimum Cycles to Failure for Level 1
    Note: SMYS is in units of ksi
    """
    # Error handling for inputs
    if od <= 0 or t <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if restraint_condition.replace("-", " ").replace(chr(8211), " ").strip().lower() not in [r.lower() for r in restraint_types]:
        raise ValueError("Invalid restraint type")
    try:
        # Check for empty or invalid restraint condition
        if not restraint_condition or restraint_condition.strip().lower() == "error":
            raise ValueError("Invalid restraint condition")

        restraint_condition = restraint_condition.strip().lower()

        if restraint_condition in ["unrestrained", "shallow restrained", "deep restrained"]:
            return _get_a16_min_l1(restraint_condition.title(), od, t, closest_pmean, pili, smys,
                                 us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                 usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                 dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
        
        elif "mixed" in restraint_condition:
            result_unres = _get_a16_min_l1("Unrestrained", od, t, closest_pmean, pili, smys,
                                         us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                         usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                         dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            
            result_res = 0
            
            if "shallow" in restraint_condition:
                result_res = _get_a16_min_l1("Shallow Restrained", od, t, closest_pmean, pili, smys,
                                           us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                           usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                           dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            elif "deep" in restraint_condition:
                result_res = _get_a16_min_l1("Deep Restrained", od, t, closest_pmean, pili, smys,
                                           us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                           usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                           dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            
            # Check if either result is an error string
            if isinstance(result_unres, str) or isinstance(result_res, str):
                raise ValueError("Error in calculation")
                
            return min(result_unres, result_res)
        else:
            raise ValueError("Invalid restraint condition")
            
    except Exception as e:
        raise ValueError(f"Error in calculation: {str(e)}")
    
def _get_md24_unbinned(OD: float, WT: float, SMYS: float, dent_type: str, 
                     rainflow_data: list, ili_pressure: float, km_params: list, 
                     bs_sd: float = 0, fatigue_curves_path: str = os.path.join(tables_folder, table_fatigue)) -> dict:
    """
    MD-2-4 Unbinned damage calculation
    
    Parameters:
        OD: Outer diameter
        WT: Wall thickness
        SMYS: SMYS value
        dent_type: Type of dent
        rainflow_data: List of [pressure_range, pressure_mean, cycles] data
        ili_pressure: ILI Pressure as %SMYS
        km_params: List of 24 KM parameters in order:
                    (lax30_us, lax75_us, lax85_us,
                    aax30_us, aax75_us, aax85_us,
                    lax30_ds, lax75_ds, lax85_ds,
                    aax30_ds, aax75_ds, aax85_ds,
                    ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                    ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                    ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                    ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw)
        bs_sd: Standard deviations for BS curve
        fatigue_curves_path: Path to the fatigue curves JSON file
    
    Returns:
        Dictionary of damage values organized by curve type and class, or error string
    """
    # Error handling for inputs
    if OD <= 0 or WT <= 0:
        raise ValueError("OD and WT must be greater than zero")
    if len(rainflow_data) == 0:
        raise ValueError("No rainflow data provided.")
    if len(km_params) != 24:
        raise ValueError("KM parameters list must contain exactly 24 values.")
    try:
        
        # Load fatigue curve parameters from JSON
        with open(fatigue_curves_path, 'r') as f:
            fatigue_curves = json.load(f)
        
        # Extract KM parameters
        (lax30_us, lax75_us, lax85_us,
        aax30_us, aax75_us, aax85_us,
        lax30_ds, lax75_ds, lax85_ds,
        aax30_ds, aax75_ds, aax85_ds,
        ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
        ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
        ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
        ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw) = km_params
        # (ah40, ah36, ah35, aj38, aj34, aj33, am40, am36, am35, ao38, ao34, ao33,
        #  am50, ao48, am48, ah50, aj48, ah48, am64, ao62, am62, ah64, aj62, ah62, k62) = km_params
        
        # %SMYS conversion factor
        factor = 100.0 * (OD / (2.0 * WT)) / SMYS

        # Initialize damage dictionary
        damage_results = {
            "ABS": {},
            "DNV": {},
            "BS": {}
        }
        
        # Initialize damage for each curve class
        for curve_data in fatigue_curves["ABS"]:
            damage_results["ABS"][curve_data["Curve"]] = 0.0
            
        for curve_data in fatigue_curves["DNV"]:
            damage_results["DNV"][curve_data["Curve"]] = 0.0
            
        for curve_data in fatigue_curves["BS"]:
            damage_results["BS"][curve_data["Curve"]] = 0.0
        
        # Process each rainflow cycle
        for row in rainflow_data:
            if len(row) < 3:
                continue
                
            pr = float(row[0])  # Pressure Range (psi)
            pm = float(row[1])  # Pressure Mean (psi)
            cycles = float(row[2])  # Number of cycles
            
            # Convert to %SMYS
            prange_pct = pr * factor
            pmean_pct = pm * factor
            pmin_pct = (pm - 0.5 * pr) * factor
            pmax_pct = (pm + 0.5 * pr) * factor
            
            # Calculate Km using the _get_km function (placeholder - you'll need to implement this)
            km = _get_km(OD, WT, dent_type,
                        lax30_us, lax75_us, lax85_us,
                        aax30_us, aax75_us, aax85_us,
                        lax30_ds, lax75_ds, lax85_ds,
                        aax30_ds, aax75_ds, aax85_ds,
                        ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                        ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                        ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                        ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                        pmean_pct, ili_pressure, prange_pct / 100, pmean_pct / 100, pmax_pct / 100)
            
            if isinstance(km, str):  # Error in km calculation
                continue
                
            # Peak Stress Range (ksi)
            s_ksi = km * (prange_pct / 100) * SMYS / 1000
            
            # Convert to MPa for DNV and BS calculations
            s_mpa = s_ksi * 6.89476  # Convert from ksi to MPa (1 ksi = 6.89476 MPa)
            
            # ABS Damage Calculations
            for curve_data in fatigue_curves["ABS"]:
                curve_name = curve_data["Curve"]
                a_ksi = curve_data["A_ksi"]
                m = curve_data["m"]
                c_ksi = curve_data["C_ksi"]
                r = curve_data["r"]
                seq_ksi = curve_data["Seq_ksi"]
                
                if s_ksi < seq_ksi:
                    cycles_to_failure = c_ksi * (s_ksi ** (-r))
                else:
                    cycles_to_failure = a_ksi * (s_ksi ** (-m))
                
                if cycles_to_failure > 0:
                    damage_results["ABS"][curve_name] += cycles / cycles_to_failure
            
            # DNV Damage Calculations
            for curve_data in fatigue_curves["DNV"]:
                curve_name = curve_data["Curve"]
                m1 = curve_data["m1"]
                loga1 = curve_data["loga1"]
                m2 = curve_data["m2"]
                loga2 = curve_data["loga2"]
                seq_mpa = curve_data["Seq"]
                
                if s_mpa > seq_mpa:
                    exp_a = loga1 - m1 * math.log10(s_mpa)
                else:
                    exp_a = loga2 - m2 * math.log10(s_mpa)
                
                cycles_to_failure = 10 ** exp_a
                
                if cycles_to_failure > 0:
                    damage_results["DNV"][curve_name] += cycles / cycles_to_failure
            
            # BS Damage Calculations
            for curve_data in fatigue_curves["BS"]:
                curve_name = curve_data["Curve"]
                log10c0 = curve_data["log10C0"]
                m = curve_data["m"]
                sd = curve_data["SD"]
                soc = curve_data["Soc"]
                
                # Use d = 2 as standard deviation factor (common practice)
                d = bs_sd
                
                if s_mpa > soc:
                    cycles_to_failure = 10 ** (log10c0 - d * sd - m * math.log10(s_mpa))
                    
                    if cycles_to_failure > 0:
                        damage_results["BS"][curve_name] += cycles / cycles_to_failure
        
        return damage_results
        
    except Exception as e:
        raise ValueError(f"Error in calculation: {str(e)}")
    
def _get_km(od: float, wt: float, restraint_condition: str, 
           lax30_us: float, lax75_us: float, lax85_us: float,
           aax30_us: float, aax75_us: float, aax85_us: float,
           lax30_ds: float, lax75_ds: float, lax85_ds: float,
           aax30_ds: float, aax75_ds: float, aax85_ds: float,
           ltr75_us_cw: float, atr75_us_cw: float, ltr85_us_cw: float,
           ltr75_us_ccw: float, atr75_us_ccw: float, ltr85_us_ccw: float,
           ltr75_ds_cw: float, atr75_ds_cw: float, ltr85_ds_cw: float,
           ltr75_ds_ccw: float, atr75_ds_ccw: float, ltr85_ds_ccw: float,
           mean_pct_psmys_raw: float, ili_pct_psmys_raw: float,
           pr_over: float, pm_over: float, px_over: float,
           tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Compute per-side K_M (US/DS) and return the max
    """
    # Error handling for inputs
    if od <= 0 or wt <= 0:
        raise ValueError("OD and WT must be greater than zero")
    try:
        # Load tables from JSON
        with open(tables_path, 'r') as f:
            data = json.load(f)
        
        # Check for empty or invalid restraint condition
        if not restraint_condition or restraint_condition.strip().lower() == "error":
            raise ValueError("Invalid restraint condition")

        restraint_condition = restraint_condition.strip().lower()

        is_restrained = (restraint_condition != "unrestrained")

        # Deciles for lambda lookup
        mean_trunc = 10 * int(mean_pct_psmys_raw / 10)
        if mean_trunc < 10:
            mean_trunc = 10
        if mean_trunc > 70:
            mean_trunc = 70

        ili_round = 10 * int(ili_pct_psmys_raw / 10 + 0.5)
        if ili_round < 10:
            ili_round = 10
        if ili_round > 70:
            ili_round = 70

        # Lambda scale factors (1 if restrained)
        if is_restrained:
            lam1 = lam2 = 1.0
        else:
            # Get coefficients lambda1 and lambda2 from Table 9/10
            table = data["9_10"]

            lam1 = lam2 = None
            # Search through 9_10 table for matching values
            for row in table:
                if (row["TMP"] == mean_trunc and row["RILIP"] == ili_round):
                    lam1 = row["lamb1"]
                    lam2 = row["lamb2"]
                    break

            if lam1 is None or lam2 is None:
                raise ValueError("Out of range, no matching lambda values found")
        
        # Aspect Ratios per side
        den85 = lax85_us + lax85_ds
        if den85 == 0:
            raise ValueError("Invalid input: sum of lax85_us and lax85_ds cannot be zero")
        
        ar_us = (ltr85_us_cw + ltr85_us_ccw) / den85
        ar_ds = (ltr85_ds_cw + ltr85_ds_ccw) / den85
        
        # Per-side K_M calculation
        km_us = _km_for_side(data, is_restrained, lam1, lam2, od, wt,
                           lax30_us, lax75_us, lax85_us, aax30_us, aax75_us, aax85_us,
                           ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                           ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                           ar_us, pr_over, pm_over, px_over)
        
        km_ds = _km_for_side(data, is_restrained, lam1, lam2, od, wt,
                           lax30_ds, lax75_ds, lax85_ds, aax30_ds, aax75_ds, aax85_ds,
                           ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                           ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                           ar_ds, pr_over, pm_over, px_over)
        
        if km_us == -1e99 and km_ds == -1e99:
            raise ValueError("Error in calculation")
        
        return max(km_us, km_ds)
        
    except Exception as e:
        raise ValueError(f"Error in calculation: {str(e)}")
    
def _get_km_l2(od: float, wt: float, restraint: str, 
            lax30_us: float, lax75_us: float, lax85_us: float,
            aax30_us: float, aax75_us: float, aax85_us: float,
            lax30_ds: float, lax75_ds: float, lax85_ds: float,
            aax30_ds: float, aax75_ds: float, aax85_ds: float,
            ltr75_us_cw: float, atr75_us_cw: float, ltr85_us_cw: float,
            ltr75_us_ccw: float, atr75_us_ccw: float, ltr85_us_ccw: float,
            ltr75_ds_cw: float, atr75_ds_cw: float, ltr85_ds_cw: float,
            ltr75_ds_ccw: float, atr75_ds_ccw: float, ltr85_ds_ccw: float,
            mean_pct_psmys_raw: float, ili_pct_psmys_raw: float,
            pr_over: float, pm_over: float, px_over: float,
            tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Wrapper for _get_km to handle "Mixed" restraint condition for Level 2
    Returns the max Km value.
    """
    # Error handling for inputs
    if od <= 0 or wt <= 0:
        raise ValueError("OD and WT must be greater than zero")
    try:
        restraint = restraint.strip().lower()

        if restraint in ["unrestrained", "shallow restrained", "deep restrained"]:
            return _get_km(od, wt, restraint, 
                            lax30_us, lax75_us, lax85_us,
                            aax30_us, aax75_us, aax85_us,
                            lax30_ds, lax75_ds, lax85_ds,
                            aax30_ds, aax75_ds, aax85_ds,
                            ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                            ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                            ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                            ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                            mean_pct_psmys_raw, ili_pct_psmys_raw,
                            pr_over, pm_over, px_over)
        elif "mixed" in restraint:
            result_unres = _get_km(od, wt, "Unrestrained", 
                                lax30_us, lax75_us, lax85_us,
                                aax30_us, aax75_us, aax85_us,
                                lax30_ds, lax75_ds, lax85_ds,
                                aax30_ds, aax75_ds, aax85_ds,
                                ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                                ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                                ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                                ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                                mean_pct_psmys_raw, ili_pct_psmys_raw,
                                pr_over, pm_over, px_over)
            result_res = 0
            
            if "shallow" in restraint:
                result_res = _get_km(od, wt, "Shallow Restrained", 
                                lax30_us, lax75_us, lax85_us,
                                aax30_us, aax75_us, aax85_us,
                                lax30_ds, lax75_ds, lax85_ds,
                                aax30_ds, aax75_ds, aax85_ds,
                                ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                                ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                                ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                                ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                                mean_pct_psmys_raw, ili_pct_psmys_raw,
                                pr_over, pm_over, px_over)
            elif "deep" in restraint:
                result_res = _get_km(od, wt, "Deep Restrained", 
                                lax30_us, lax75_us, lax85_us,
                                aax30_us, aax75_us, aax85_us,
                                lax30_ds, lax75_ds, lax85_ds,
                                aax30_ds, aax75_ds, aax85_ds,
                                ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                                ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                                ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                                ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                                mean_pct_psmys_raw, ili_pct_psmys_raw,
                                pr_over, pm_over, px_over)
                
            return max(result_unres, result_res)
        else:
            raise ValueError("Invalid restraint condition")
            
    except Exception as e:
        raise ValueError(f"Error in calculation: {str(e)}")
    
def _km_for_side(data: dict, is_restrained: bool, lam1: float, lam2: float,
                od: float, wt: float, lax30: float, lax75: float, lax85: float,
                aax30: float, aax75: float, aax85: float,
                ltr75_cw: float, atr75_cw: float, ltr85_cw: float,
                ltr75_ccw: float, atr75_ccw: float, ltr85_ccw: float,
                ar_side: float, pr_over: float, pm_over: float, px_over: float) -> float:
    """
    One side (US or DS): compute AR bin, pull b's, evaluate CW & CCW, take max
    """
    # Error handling for inputs
    if od <= 0 or wt <= 0:
        raise ValueError("OD and WT must be greater than zero")
    try:
        # Determine the b coefficients based on restraint and AR bin
        b = None
        # Search through 9_10 table for matching values
        if is_restrained:
            table = data["29"]
        else:
            table = data["30"]
        for row in table:
            if (row["AR_LB"] <= ar_side < row["AR_UB"]):
                b = list(row.values())[2:]  # All b coefficients start from index 2
                break

        # Calculate KM for CW and CCW
        km_cw = _km_for_pair(is_restrained, lam1, lam2, od, wt,
                           lax30, lax75, lax85, aax30, aax75, aax85,
                           ltr75_cw, atr75_cw, ltr85_cw,
                           pr_over, pm_over, px_over, b)
        
        km_ccw = _km_for_pair(is_restrained, lam1, lam2, od, wt,
                            lax30, lax75, lax85, aax30, aax75, aax85,
                            ltr75_ccw, atr75_ccw, ltr85_ccw,
                            pr_over, pm_over, px_over, b)
        
        if km_cw == -1e99 and km_ccw == -1e99:
            return -1e99
        else:
            return max(km_cw, km_ccw)
            
    except Exception:
        return -1e99
    
def _km_for_pair(is_restrained: bool, lam1: float, lam2: float,
                od: float, wt: float, lax30: float, lax75: float, lax85: float,
                aax30: float, aax75: float, aax85: float,
                ltr75: float, atr75: float, ltr85: float,
                pr_over: float, pm_over: float, px_over: float, b: list) -> float:
    """
    One pairing (e.g., CW) for a given side
    """
    try:
        if wt <= 0 or od <= 0:
            return -1e99
        
        if is_restrained:
            # Eq. (4): restrained
            if (lax30 <= 0 or lax75 <= 0 or ltr75 <= 0 or 
                aax30 <= 0 or aax75 <= 0 or atr75 <= 0):
                return -1e99
            
            x1 = (aax30 * aax75) / (lax30 * lax75 * od * wt * 0.001) * (ltr75 / lax75) ** 0.5
            x2 = ((aax75 * atr75) / (lax75 * ltr75 * wt ** 2)) ** 0.5
            xbar1 = (x1 - 2.97593) / 4.02113
            xbar2 = (x2 - 0.22786) / 0.16693
            
            return _km_restrained_from_b(od, wt, xbar1, xbar2, pr_over, pm_over, px_over, b)
        else:
            # Eq. (8): unrestrained
            if (lax30 <= 0 or lax75 <= 0 or lax85 <= 0 or ltr85 <= 0 or
                aax30 <= 0 or aax75 <= 0 or aax85 <= 0):
                return -1e99
            
            x1 = lam1 * (math.sqrt(aax85 * aax75) / (lax85 * lax75)) * (lax85 / ltr85) ** 0.25
            x2 = lam2 * (aax30 / (lax30 * wt)) ** 0.25
            xbar1 = (x1 - 0.01478) / 0.014801
            xbar2 = (x2 - 0.67486) / 0.10759
            
            return _km_unrestrained_from_b(od, wt, xbar1, xbar2, pr_over, pm_over, px_over, b)
            
    except Exception:
        return -1e99
    
def _km_restrained_from_b(od: float, wt: float, xbar1: float, xbar2: float,
                        pr: float, pm: float, px: float, b: list) -> float:
    """
    KM assembly (Restrained, Eq. 2): product of two sums
    """
    try:
        geom = (od / wt) / 100
        c = _build_c_from_b(b, True, geom, pr, pm, px)
        
        if not c or len(c) < 11:
            return -1e99
        
        sum_a = (abs(c[1]) + 
                abs(c[2]) * (xbar1 + c[3]) ** 2 + 
                abs(c[4]) * (xbar2 + c[5]) ** 2)
        
        sum_b = (abs(c[6]) + 
                abs(c[7]) * math.exp(-((xbar1 + c[8]) ** 2)) + 
                abs(c[9]) * math.exp(-((xbar2 + c[10]) ** 2)))
        
        return sum_a * sum_b
        
    except Exception:
        return -1e99

def _km_unrestrained_from_b(od: float, wt: float, xbar1: float, xbar2: float,
                          pr: float, pm: float, px: float, b: list) -> float:
    """
    KM assembly (Unrestrained, Eq. 6): product of two sums
    """
    try:
        geom = (od / wt) / 100
        c = _build_c_from_b(b, False, geom, pr, pm, px)
        
        if not c or len(c) < 11:
            return -1e99
        
        sum_a = (abs(c[1]) + 
                abs(c[2]) * (xbar1 + c[3]) ** 2 + 
                abs(c[4]) * (xbar2 + c[5]) ** 2)
        
        sum_b = (abs(c[6]) + 
                abs(c[7]) * math.exp(-((xbar1 + c[8]) ** 2)) + 
                abs(c[9]) * math.exp(-((xbar2 + c[10]) ** 2)))
        
        return sum_a * sum_b
        
    except Exception:
        return -1e99
    
def _build_c_from_b(b: list, is_restrained: bool, geom: float, 
                   pr: float, pm: float, px: float) -> list:
    """
    Build c1..c10 from b (Table 29/30)
    """
    try:
        if not b:
            return []
        
        c = [0] * 11  # c[0] unused, c[1] to c[10]
        
        if is_restrained:
            # Table 29: 5 numbers per c_n (b1..b50)
            if len(b) < 50:
                return []
            
            for n in range(1, 11):
                k = 5 * (n - 1)
                c[n] = (b[k] + b[k+1] * geom + b[k+2] * pr + 
                       b[k+3] * pm + b[k+4] * px)
        else:
            # Table 30: 4 numbers per c_n (b1..b40)
            if len(b) < 40:
                return []
            
            for n in range(1, 11):
                k = 4 * (n - 1)
                c[n] = (b[k] + b[k+1] * geom + b[k+2] * pr + b[k+3] * pm)
        
        return c
        
    except Exception:
        return []
    
def _get_scale_factor_md53(certainty: float, safety_factor: float, criteria: str, level: str, 
                          tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Get the scale factor for Level 0 through 2 using Table D.1 through D.7 from MD-5-3
    """
    with open(tables_path, 'r') as f:
        data = json.load(f)
    try:
        table = data["D.1_6"]

        sf = None
        # Search through D.1_6 table for matching values
        # If Criteria is "Multiple", treat it as "Corrosion"
        if criteria.strip().lower() == "multiple" or criteria.strip().lower() == "metal loss" or criteria.strip().lower() == "corrosion":
            criteria = "Corrosion"

        # Make sure that Certainty is a decimal value (e.g., 0.9, 0.8, 0.7)
        if certainty > 1.0:
            certainty = certainty / 100.0

        # If level is "1", treat it as "2"
        if level == "1":
            level = "2"

        for row in table:
            if (str(row["Level"]) == level and 
                row["Interaction"] == criteria and 
                row["Safety_Factor"] == safety_factor and
                row["Certainty"] == certainty):
                sf = row["Scale_Factor"]
                break

        if sf is None:
            return "Out of Range"
        
        return sf
    
    except Exception:
        return "Error in calculation"

def _get_scale_factor_l2(restraint: str, certainty: float, safety_factor: float, 
                       criteria: str, metal_loss_location: str = math.nan, weld_interaction_sf: int = math.nan,
                       tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Get the scale factor for Level 2 only using Tables 15 through 18 from MD-2-4
    """
    with open(tables_path, 'r') as f:
        data = json.load(f)
    try:
        table = data["15_28"]

        sf = None
        # Search through 15_28 table for matching values
        # If Criteria is "Multiple", treat it as "Corrosion" and ignore weld_interaction_sf
        if criteria.strip().lower() == "multiple" or criteria.strip().lower() == "metal loss" or criteria.strip().lower() == "corrosion":
            criteria = "Corrosion"
            weld_interaction_sf = math.nan  # Ignore weld interaction SF

        if criteria.strip().lower() == "plain":
            weld_interaction_sf = math.nan  # Ignore weld interaction SF
            metal_loss_location = math.nan

        if criteria.strip().lower() == "weld":
            metal_loss_location = math.nan  # Ignore metal loss location

        # Make sure that Certainty is a decimal value (e.g., 0.9, 0.8, 0.7)
        if certainty > 1.0:
            certainty = certainty / 100.0

        # If metal_loss_location is not provided, set it to NaN
        if not metal_loss_location or metal_loss_location == "":
            metal_loss_location = math.nan

        # Check the Restraint input and convert it to either "Restrained" or "Unrestrained" based on substring
        # Treat Shallow/Deep Restrained as "Restrained"
        if "unrestrained" in restraint.strip().lower():
            restraint = "Unrestrained"
        else:
            restraint = "Restrained"

        for row in table:
            if (row["Interaction"] == criteria and 
                ((pd.isna(row["OD_ID"]) and pd.isna(metal_loss_location)) or (row["OD_ID"] == metal_loss_location)) and
                ((pd.isna(row["Reduction_Factor"]) and pd.isna(weld_interaction_sf)) or (row["Reduction_Factor"] == weld_interaction_sf)) and
                row["Restraint"] == restraint and
                row["Safety_Factor"] == safety_factor and
                row["Certainty"] == certainty):
                sf = row["Scale_Factor"]
                break

        if sf is None:
            return "Out of Range"
        
        return sf
    
    except Exception:
        traceback.print_exc()
        return "Error in calculation"
    
def _get_scale_factor_l2_wrap(restraint: str, certainty: float, safety_factor: float, 
                            criteria: str, metal_loss_location: str = math.nan, weld_interaction_sf: int = math.nan,
                            tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Wrapper function for _get_scale_factor_l2 to repeat calculations if Restraint is "Mixed"
    Return the maximum scale factor from both restraint conditions
    Treat Shallow/Deep Restrained as "Restrained"
    """
    if "mixed" in restraint.strip().lower():
        sf1 = _get_scale_factor_l2("Restrained", certainty, safety_factor, criteria, metal_loss_location, weld_interaction_sf, tables_path)
        sf2 = _get_scale_factor_l2("Unrestrained", certainty, safety_factor, criteria, metal_loss_location, weld_interaction_sf, tables_path)
        return max(sf1, sf2)
    else:
        return _get_scale_factor_l2(restraint, certainty, safety_factor, criteria, metal_loss_location, weld_interaction_sf, tables_path)

def get_restraint(quad_values: list, od: float, dent_depth_pct: float) -> str:
    """
    Determine the Restraint string based on the maximum RP value and Shallow/Deep condition
    
    Args:
        quad_values: List of quadrant values
        od: Outer diameter, inches
        dent_depth_pct: Dent depth percentage, %
    
    Returns:
        Restraint condition string or "Error"
    """
    try:
        # Input validation
        if not quad_values or not all([od, dent_depth_pct]):
            return "Error"
        
        # Filter out NaN values when calculating max
        valid_values = []
        for value in quad_values:
            if pd.notna(value) and isinstance(value, (int, float)):
                valid_values.append(value)
        
        if not valid_values:
            return "Error"
        
        max_value = max(valid_values)
        
        # Determine if shallow or deep based on OD
        if od <= 12.75:
            is_shallow = dent_depth_pct < 4
        else:  # od > 12.75
            is_shallow = dent_depth_pct < 2.5
        
        # Additional validation
        if not isinstance(max_value, (int, float)) or max_value == 0:
            return "Error"
        
        # Determine restraint condition based on max_value and depth
        if max_value < 15:
            return "Unrestrained"
        elif max_value > 25:
            if is_shallow:
                return "Shallow Restrained"
            else:
                return "Deep Restrained"
        elif 15 <= max_value <= 25:
            if is_shallow:
                return "Shallow Mixed"
            else:
                return "Deep Mixed"
        else:
            return "Error"
            
    except Exception:
        return "Error"
    
def _get_criteria_type(metal_loss_interaction: bool, weld_interaction: bool) -> str:
    """
    Determine the criteria type based on metal loss and weld interaction flags
    
    Args:
        metal_loss_interaction: Boolean indicating if there's metal loss interaction
        weld_interaction: Boolean indicating if there's weld interaction
    
    Returns:
        Criteria type string: "Metal Loss", "Weld", "Plain", or "Multiple"
    """
    if metal_loss_interaction and not weld_interaction:
        return "Metal Loss"
    elif not metal_loss_interaction and weld_interaction:
        return "Weld"
    elif not metal_loss_interaction and not weld_interaction:
        return "Plain"
    else:  # Both are True
        return "Multiple"
    
def _get_RL(damage: float, service_years: float, scale_factor: float, weld_sf: float, ml_rf: float) -> tuple[float, float]:
    """
    Calculate the Remaining Life (RL) and RL with Factors
    """
    RL = RL_sf = None
    if isinstance(scale_factor, (int, float)):
        RL = (1.0 - damage) / (damage / service_years)
        RL_sf = RL / (scale_factor * weld_sf * ml_rf)

    if not RL or not RL_sf:
        return (math.nan, math.nan)
    
    return (RL, RL_sf)

def _get_total_damage(dd: cc.DentCycles, km_values: list, prange_list: list, MD49_cycles: list, curve_selection: dict, fatigue_curves_path: str = os.path.join(tables_folder, table_fatigue)) -> float:
    """
    Calculate total damage for all fatigue curve options.
    Args:
        dd: DentCycles object containing pipe and feature characteristics
        km_values: List of K_M values for each bin
        prange_list: List of pressure ranges (as percentages of SMYS) for each bin
        MD49_cycles: List of cycle counts for each bin
        curve_selection: Dictionary containing fatigue category and curve selection. If using BS, also includes SD value.
        fatigue_curves_path: Path to the JSON file containing fatigue curve parameters
    Returns:
        Total damage as a float or error string
    """
    try:
        # Make sure length of km_values, prange_list, and MD49_cycles are the same
        if not (len(km_values) == len(prange_list) == len(MD49_cycles)):
            raise ValueError("Input lists must have the same length")
        
        # Load fatigue curve parameters from JSON
        with open(fatigue_curves_path, 'r') as f:
            fatigue_curves = json.load(f)
        
        press_range_list = [(val/100) * dd.SMYS * 2 * dd.WT / dd.OD for val in prange_list]  # Pressure range in psi
        damage_results = 0.0
        
        # ABS Damage Calculations
        if curve_selection["Category"] == "ABS":
            for curve in fatigue_curves[curve_selection["Category"]]:
                if curve["Curve"] == curve_selection["Curve"]:
                    curve_data = curve
                    break
            a_ksi = curve_data["A_ksi"]
            m = curve_data["m"]
            c_ksi = curve_data["C_ksi"]
            r = curve_data["r"]
            seq_ksi = curve_data["Seq_ksi"]

            for i in range(len(km_values)):
                # Peak Stress Range (ksi)
                s_ksi = km_values[i] * (press_range_list[i] * dd.OD / (2 * dd.WT)) / 1000  # Convert psi to ksi (1 ksi = 1000 psi)
                
                if s_ksi < seq_ksi:
                    cycles_to_failure = c_ksi * (s_ksi ** (-r))
                else:
                    cycles_to_failure = a_ksi * (s_ksi ** (-m))
                
                if cycles_to_failure > 0:
                    damage_results += MD49_cycles[i] / cycles_to_failure
        
        # DNV Damage Calculations
        if curve_selection["Category"] == "DNV":
            for curve in fatigue_curves[curve_selection["Category"]]:
                if curve["Curve"] == curve_selection["Curve"]:
                    curve_data = curve
                    break
            m1 = curve_data["m1"]
            loga1 = curve_data["loga1"]
            m2 = curve_data["m2"]
            loga2 = curve_data["loga2"]
            seq_mpa = curve_data["Seq"]

            for i in range(len(km_values)):
                # Peak Stress Range (ksi)
                s_ksi = km_values[i] * (press_range_list[i] * dd.OD / (2 * dd.WT)) / 1000  # Convert psi to ksi (1 ksi = 1000 psi)

                # Convert to MPa for DNV and BS calculations
                s_mpa = s_ksi * 6.89476  # Convert from ksi to MPa (1 ksi = 6.89476 MPa)
                
                if s_mpa > seq_mpa:
                    exp_a = loga1 - m1 * math.log10(s_mpa)
                else:
                    exp_a = loga2 - m2 * math.log10(s_mpa)
                
                cycles_to_failure = 10 ** exp_a
                
                if cycles_to_failure > 0:
                    damage_results += MD49_cycles[i] / cycles_to_failure
        
        # BS Damage Calculations
        if curve_selection["Category"] == "BS":
            for curve in fatigue_curves[curve_selection["Category"]]:
                if curve["Curve"] == curve_selection["Curve"]:
                    curve_data = curve
                    break
            log10c0 = curve_data["log10C0"]
            m = curve_data["m"]
            sd = curve_data["SD"]
            soc = curve_data["Soc"]
            
            # Use d = 2 as standard deviation factor (common practice)
            d = curve_selection["SD"] if "SD" in curve_selection else 2

            for i in range(len(km_values)):
                # Peak Stress Range (ksi)
                s_ksi = km_values[i] * (press_range_list[i] * dd.OD / (2 * dd.WT)) / 1000  # Convert psi to ksi (1 ksi = 1000 psi)

                # Convert to MPa for DNV and BS calculations
                s_mpa = s_ksi * 6.89476  # Convert from ksi to MPa (1 ksi = 6.89476 MPa)
            
                if s_mpa > soc:
                    cycles_to_failure = 10 ** (log10c0 - d * sd - m * math.log10(s_mpa))
                    
                if cycles_to_failure > 0:
                    damage_results += MD49_cycles[i] / cycles_to_failure

        return damage_results
    except Exception:
        traceback.print_exc()
        raise ValueError("Error in calculation")


def get_fatigue_life(dd: cc.DentCycles, pf: mpf.DentProfiles, cycles: np.ndarray, curve_selection: dict, 
            confidence: float=0.8,
            interaction_corrosion: bool=False,
            interaction_weld: bool=False,
            CPS: bool=False,
            ili_pressure: float=None,
            dent_depth_percent: float=None,
            ml_depth_percent: float=None) -> tuple[str, dict, dict]:
    """
    Compute the fatigue life using pipe characteristics, rainflow data, and feature sizing from MD-4-9 profiles.
    Depending on the available inputs, performs the assessment for each level of analysis.
    Parameters
    ----------
    dd : DentCycles
        DentCycles object containing pipe and feature characteristics
    pf : DentProfiles
        DentProfiles object containing MD-4-9 profile results
    cycles : np.ndarray
        Array of rainflow cycles with columns: [Pressure Range (psig), Pressure Mean (psig), Cycle Count, Index Start, Index End]
    curve_selection : dict
        Dictionary containing curve selection parameters
    press_dict : dict
        Dictionary containing pressure bin values (pmin, pmax, prange, pmean) in %SMYS. Default is the standard MD-4-9 bins.
    confidence : float, optional
        Confidence level for scale factor determination, default is 0.8 (80%).
    interaction_corrosion : bool, optional
        Flag indicating if there is metal loss interaction. Default is False.
    interaction_weld : bool, optional
        Flag indicating if there is weld interaction. Default is False.
    CPS : bool, optional
        Flag indicating if the pipeline is CPS. Default is False.
    ili_pressure : float, optional
        Pipeline pressure during the ILI assessment, psig. This is required to perform Level 1 and Level 2 analyses.
    dent_depth_percent : float, optional
        Dent depth percentage, %OD. If NaN or None, use the calculated value from pf DentProfiles.
    ml_depth_percent : float, optional
        Metal loss depth percentage, %WT. If NaN or None, assume no metal loss.
    Returns
    -------
    calc_restraint : str
        Restraint condition determined from the profile results and dent depth.
    rp_results : dict
        Dictionary containing the restraint parameter results for each quadrant.
    fatigue_results : dict
        Dictionary containing the damage results for each level of analysis and with or without scale factors.
    """

    try:
        # Error handling for inputs
        if not isinstance(dd, cc.DentCycles) or not isinstance(pf, mpf.DentProfiles):
            raise ValueError("Invalid DentCycles or DentProfiles object")
        # If dent_depth_percent is NaN or None, use the calculated value from pf._dent_depth_percent. Provide a warning.
        if pd.isna(dent_depth_percent) or dent_depth_percent is None:
            dent_depth_percent = pf._dent_depth_percent
            print(f"Warning: Dent depth percentage was missing. Using calculated value of {dent_depth_percent:.3f} %OD from profile data.")

        # Separate the pressure bins into lists
        pmin_list = [val["pmin"] for val in dd._cycles_binned.values()]
        pmax_list = [val["pmax"] for val in dd._cycles_binned.values()]
        prange_list = [val["prange"] for val in dd._cycles_binned.values()]
        pmean_list = [val["pmean"] for val in dd._cycles_binned.values()]
        cycles_binned = [val["cycle_count"] for val in dd._cycles_binned.values()]

        # Perform calculations and populate results
        max_pressure_mean = dd._mean_max
        pressure_mean = dd._mean_avg
        
        pmean_list_short = [25, 45, 65]
        closest_pmean = min(pmean_list_short, key=lambda x: abs(x - pressure_mean))

        rp_US_CCW = mpf.get_restraint_parameter(pf._results_axial_us["areas"][15],
                                                 pf._results_circ_us_ccw["areas"][15],
                                                 pf._results_circ_us_ccw["lengths"][70]["length"],
                                                 pf._results_axial_us["lengths"][15]["length"],
                                                 pf._results_axial_us["lengths"][30]["length"],
                                                 pf._results_axial_us["lengths"][50]["length"],
                                                 pf._results_circ_us_ccw["lengths"][80]["length"])
        rp_US_CW = mpf.get_restraint_parameter(pf._results_axial_us["areas"][15],
                                                 pf._results_circ_us_cw["areas"][15],
                                                 pf._results_circ_us_cw["lengths"][70]["length"],
                                                 pf._results_axial_us["lengths"][15]["length"],
                                                 pf._results_axial_us["lengths"][30]["length"],
                                                 pf._results_axial_us["lengths"][50]["length"],
                                                 pf._results_circ_us_cw["lengths"][80]["length"])
        rp_DS_CCW = mpf.get_restraint_parameter(pf._results_axial_ds["areas"][15],
                                                 pf._results_circ_ds_ccw["areas"][15],
                                                 pf._results_circ_ds_ccw["lengths"][70]["length"],
                                                 pf._results_axial_ds["lengths"][15]["length"],
                                                 pf._results_axial_ds["lengths"][30]["length"],
                                                 pf._results_axial_ds["lengths"][50]["length"],
                                                 pf._results_circ_ds_ccw["lengths"][80]["length"])
        rp_DS_CW = mpf.get_restraint_parameter(pf._results_axial_ds["areas"][15],
                                                 pf._results_circ_ds_cw["areas"][15],
                                                 pf._results_circ_ds_cw["lengths"][70]["length"],
                                                 pf._results_axial_ds["lengths"][15]["length"],
                                                 pf._results_axial_ds["lengths"][30]["length"],
                                                 pf._results_axial_ds["lengths"][50]["length"],
                                                 pf._results_circ_ds_cw["lengths"][80]["length"])
        calc_restraint = get_restraint([rp_US_CCW, rp_US_CW, rp_DS_CCW, rp_DS_CW], dd.OD, dent_depth_percent)

        # Determine criteria type
        criteria = _get_criteria_type(interaction_corrosion, interaction_weld)

        # Calculate scale and safety factors
        cps_sf = 4.0 if CPS else 2.0
        weld_sf = 10.0 if interaction_weld == True else 5.0
        if not pd.isna(ml_depth_percent) and ml_depth_percent > 0:
            ml_rf = dd.WT/(dd.WT - (ml_depth_percent/100)*dd.WT)
        else:
            ml_rf = 1.0

        # Level 0 RLA Result
        l0_Km_result = _get_km_l0(dd.OD, dd.WT, calc_restraint, max_pressure_mean)
        l0_sf = _get_scale_factor_md53(confidence, cps_sf, criteria, "0")
        RLA_l0, RLA_l0_sf = _get_RL(_get_total_damage(dd, [l0_Km_result] * len(prange_list), prange_list, cycles_binned, curve_selection), dd.service_years, l0_sf, weld_sf, ml_rf)

        # Level 0.5 RLA Result
        l05_Km_results = [0] * len(prange_list)
        for i, val in enumerate(prange_list):
            l05_Km_results[i] = _get_km_l05(dd.OD, dd.WT, val, calc_restraint)
        l05_sf = _get_scale_factor_md53(confidence, cps_sf, criteria, "0.5")
        RLA_l05, RLA_l05_sf = _get_RL(_get_total_damage(dd, l05_Km_results, prange_list, cycles_binned, curve_selection), dd.service_years, l05_sf, weld_sf, ml_rf)
        
        # Level 0.5+ RLA Result
        l05p_Km_results = [0] * len(prange_list)
        for i, _ in enumerate(prange_list):
            l05p_Km_results[i] = get_km_l05p(dd.OD, dd.WT, prange_list[i], pmean_list[i], calc_restraint)
        l05p_sf = _get_scale_factor_md53(confidence, cps_sf, criteria, "0.5+")
        RLA_l05p, RLA_l05p_sf = _get_RL(_get_total_damage(dd, l05p_Km_results, prange_list, cycles_binned, curve_selection), dd.service_years, l05p_sf, weld_sf, ml_rf)

        # Level 0.75 RLA Result
        l075_Km_results = [0] * len(prange_list)
        for i, val in enumerate(prange_list):
            l075_Km_results[i] = _get_km_l075(dd.OD, dd.WT, val, dent_depth_percent, calc_restraint)
        l075_sf = _get_scale_factor_md53(confidence, cps_sf, criteria, "0.75")
        RLA_l075, RLA_l075_sf = _get_RL(_get_total_damage(dd, l075_Km_results, prange_list, cycles_binned, curve_selection), dd.service_years, l075_sf, weld_sf, ml_rf)

        # Level 0.75+ RLA Result
        l075p_Km_results = [0] * len(prange_list)
        for i, _ in enumerate(prange_list):
            l075p_Km_results[i] = _get_km_l075p(dd.OD, dd.WT, prange_list[i], pmean_list[i], dent_depth_percent, calc_restraint)
        l075p_sf = _get_scale_factor_md53(confidence, cps_sf, criteria, "0.75+")
        RLA_l075p, RLA_l075p_sf = _get_RL(_get_total_damage(dd, l075p_Km_results, prange_list, cycles_binned, curve_selection), dd.service_years, l075p_sf, weld_sf, ml_rf)

        if ili_pressure is not None:
            ili_pressure_psmys = 100 * (ili_pressure * dd.OD / (2 * dd.WT)) / dd.SMYS
            # Level 1 RLA Result
            l1_N_result = _get_n_l1(calc_restraint, dd.OD, dd.WT, closest_pmean, ili_pressure_psmys, dd.SMYS/1000, 
                                pf.US_LAX, pf.US_AAX, pf.DS_LAX, pf.DS_AAX, 
                                pf.US_CCW_LTR, pf.US_CCW_ATR, pf.US_CW_LTR, pf.US_CW_ATR,
                                pf.DS_CCW_LTR, pf.DS_CCW_ATR, pf.DS_CW_LTR, pf.DS_CW_ATR)
            if closest_pmean == 25 or closest_pmean == 45:
                damage_val = (dd._Neq_SSI / l1_N_result) * (13000/(30 * dd.SMYS / 100)) ** 3
            else:
                damage_val = (dd._Neq_SSI / l1_N_result) * (13000/(40 * dd.SMYS / 100)) ** 3
            l1_sf = _get_scale_factor_md53(confidence, cps_sf, criteria, "1")
            RLA_l1, RLA_l1_sf = _get_RL(damage_val, dd.service_years, l1_sf, weld_sf, ml_rf)

            # Level 2 RLA Results
            l2_N_results = [0] * len(prange_list)
            for i, _ in enumerate(pmin_list):
                l2_N_results[i] = _get_n_l2(calc_restraint, dd.OD, dd.WT, pmean_list[i], pmax_list[i], pmin_list[i], ili_pressure_psmys, dd.SMYS/1000,
                                        pf.US_LAX, pf.US_AAX, pf.DS_LAX, pf.DS_AAX, 
                                        pf.US_CCW_LTR, pf.US_CCW_ATR, pf.US_CW_LTR, pf.US_CW_ATR,
                                        pf.DS_CCW_LTR, pf.DS_CCW_ATR, pf.DS_CW_LTR, pf.DS_CW_ATR)
            damage_val = sum(cycles_binned[i] / l2_N_results[i] for i in range(len(l2_N_results)))
            l2_sf = _get_scale_factor_md53(confidence, cps_sf, criteria, "2")
            RLA_l2, RLA_l2_sf = _get_RL(damage_val, dd.service_years, l2_sf, weld_sf, ml_rf)

            # Level 2 MD-2-4 Binned RLA Results
            l2_Km_results = [0] * len(prange_list)
            lax30_us = pf._results_axial_us["lengths"][30]["length"]
            lax75_us = pf._results_axial_us["lengths"][75]["length"]
            lax85_us = pf._results_axial_us["lengths"][85]["length"]
            aax30_us = pf._results_axial_us["areas"][30]
            aax75_us = pf._results_axial_us["areas"][75]
            aax85_us = pf._results_axial_us["areas"][85]
            lax30_ds = pf._results_axial_ds["lengths"][30]["length"]
            lax75_ds = pf._results_axial_ds["lengths"][75]["length"]
            lax85_ds = pf._results_axial_ds["lengths"][85]["length"]
            aax30_ds = pf._results_axial_ds["areas"][30]
            aax75_ds = pf._results_axial_ds["areas"][75]
            aax85_ds = pf._results_axial_ds["areas"][85]
            ltr75_us_cw = pf._results_circ_us_cw["lengths"][75]["length"]
            atr75_us_cw = pf._results_circ_us_cw["areas"][75]
            ltr85_us_cw = pf._results_circ_us_cw["lengths"][85]["length"]
            ltr75_us_ccw = pf._results_circ_us_ccw["lengths"][75]["length"]
            atr75_us_ccw = pf._results_circ_us_ccw["areas"][75]
            ltr85_us_ccw = pf._results_circ_us_ccw["lengths"][85]["length"]
            ltr75_ds_cw = pf._results_circ_ds_cw["lengths"][75]["length"]
            atr75_ds_cw = pf._results_circ_ds_cw["areas"][75]
            ltr85_ds_cw = pf._results_circ_ds_cw["lengths"][85]["length"]
            ltr75_ds_ccw = pf._results_circ_ds_ccw["lengths"][75]["length"]
            atr75_ds_ccw = pf._results_circ_ds_ccw["areas"][75]
            ltr85_ds_ccw = pf._results_circ_ds_ccw["lengths"][85]["length"]
            for i, _ in enumerate(prange_list):
                l2_Km_results[i] = _get_km_l2(dd.OD, dd.WT, calc_restraint,
                                            lax30_us, lax75_us, lax85_us, aax30_us, aax75_us, aax85_us, 
                                            lax30_ds, lax75_ds, lax85_ds, aax30_ds, aax75_ds, aax85_ds, 
                                            ltr75_us_cw, atr75_us_cw, ltr85_us_cw, ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                                            ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw, ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                                            pmean_list[i], ili_pressure_psmys, prange_list[i]/100, pmean_list[i]/100, pmax_list[i]/100)
            l2_md24_sf = _get_scale_factor_l2_wrap(calc_restraint, confidence, cps_sf, criteria, dd.ml_location, weld_sf)
            RLA_l2_md24, RLA_l2_md24_sf = _get_RL(_get_total_damage(dd, l2_Km_results, prange_list, cycles_binned, curve_selection), dd.service_years, l2_md24_sf, weld_sf, ml_rf)

            # Level 2 MD-2-4 Unbinned RLA Results
            km_params = (lax30_us, lax75_us, lax85_us,
                        aax30_us, aax75_us, aax85_us,
                        lax30_ds, lax75_ds, lax85_ds,
                        aax30_ds, aax75_ds, aax85_ds,
                        ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                        ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                        ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                        ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw)
            l2_md24_unbinned_results = _get_md24_unbinned(dd.OD, dd.WT, dd.SMYS, calc_restraint,
                                                        cycles, ili_pressure_psmys, km_params)
            
            # Determine the Remaining Life (RL) for each fatigue curve category and curve option
            RLA_l2_md24_unbinned = {"ABS": {}, "DNV": {}, "BS": {}}
            RLA_l2_md24_unbinned_sf = {"ABS": {}, "DNV": {}, "BS": {}}
            for category in ["ABS", "DNV", "BS"]:
                for curve, damage in l2_md24_unbinned_results[category].items():
                    RLA_l2_md24_unbinned[category][curve], RLA_l2_md24_unbinned_sf[category][curve] = _get_RL(float(damage), dd.service_years, l2_md24_sf, weld_sf, ml_rf)
        
        # Build output dictionaries
        rp_values = {
                "US-CCW": rp_US_CCW,
                "US-CW": rp_US_CW,
                "DS-CCW": rp_DS_CCW,
                "DS-CW": rp_DS_CW
        }
        fatigue_life = {
                "0": {"No SF": RLA_l0, "Yes SF": RLA_l0_sf},
                "0.5": {"No SF": RLA_l05, "Yes SF": RLA_l05_sf},
                "0.5+": {"No SF": RLA_l05p, "Yes SF": RLA_l05p_sf},
                "0.75": {"No SF": RLA_l075, "Yes SF": RLA_l075_sf},
                "0.75+": {"No SF": RLA_l075p, "Yes SF": RLA_l075p_sf},
        }
        
        # Only include Level 1 and Level 2 results if ili_pressure was provided
        if ili_pressure is not None:
            fatigue_life["1"] = {"No SF": RLA_l1, "Yes SF": RLA_l1_sf}
            fatigue_life["2"] = {"No SF": RLA_l2, "Yes SF": RLA_l2_sf}
            fatigue_life["2_md24"] = {"No SF": RLA_l2_md24, "Yes SF": RLA_l2_md24_sf}
            fatigue_life["2_md24_unbinned"] = {"No SF": RLA_l2_md24_unbinned, "Yes SF": RLA_l2_md24_unbinned_sf}
        
        return (calc_restraint, rp_values, fatigue_life)

    except Exception as e:
        traceback.print_exc()
        raise e


