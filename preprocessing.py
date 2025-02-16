import numpy as np
import pandas as pd


# Categorize street names into 'Highway/Freeway', 'Local Road', or 'Other'
state_abbrs = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
    'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]

def categorize_street(street_name):

    street_name = str(street_name)  # Ensure it's a string
    
    # Check for highways
    if any(x in street_name for x in ['I-', 'US-', 'State Route', 'SR-', 
                                      'Hwy', 'Highway','Fwy', 'Expressway', 'Expy',
                                      'Turnpike', 'Pike', 'Pkwy', 
                                      'Interstate', 'Trwy', 'Tpke',
                                      'Bypass', 'Corridor']) or \
       any(f"{abbr}-" in street_name for abbr in state_abbrs):  # Detect state-specific highways
        return 'Highway/Freeway'
    
    # Check for local roads
    elif any(x in street_name for x in ['Ave', 'Rd', 'St', 'Blvd', 'Dr', 'Ct', 'Pl', 'Ln', 'Way','Trail', 'Plaza']):
        return 'Local Road'
    
    # Default category
    else:
        return 'Other'


# Encode street type into binary feature: 1 for Highway/Freeway, 0 for Local Road / Other
def encode_street_type(street_type):
    if pd.isna(street_type):  
        return 0  # Default to local road if missing
    
    street_type = str(street_type)
    
    if street_type == 'Highway/Freeway':
        return 1  # Highway/Freeway
    else:
        return 0  # Local Road / Other
    
    
def encode_traffic_signal(traffic_signal):
    if pd.isna(traffic_signal):
        return 0  # Default to no traffic signal if missing
    
    traffic_signal = str(traffic_signal)
    
    if traffic_signal == 'True':
        return 1  # Traffic signal present
    else:
        return 0  # No traffic signal


def encode_crossing(crossing):
    """
    TODO: Implement the function to encode the 'Crossing' feature.

    Instructions:
    - If the value is missing (NaN), return 0 (assume no crossing).
    - Convert the input to a string.
    - If the string is 'True', return 1 (indicating a crossing is present).
    - Otherwise, return 0 (no crossing).

    Example:
        encode_crossing('True')  -> 1
        encode_crossing('False') -> 0
        encode_crossing(None)    -> 0
    """
    pass  # TODO: Implement this function
    