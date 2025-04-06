import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.widgets import Slider

class AccidentSeverityModel(nn.Module):
    def __init__(self):
        super(AccidentSeverityModel, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = AccidentSeverityModel()
model.load_state_dict(torch.load('best_accident_severity_model.pth'))
model.eval()

# Default features 
features = {
    'Traffic_Signal_Flag': 0,
    'Crossing_Flag': 0,
    'Highway_Flag': 1,
    'Distance(mi)': 1.0,  # Default distance
    'Start_Hour': 12,  # Default hour for the slider
    'Start_Month': 6,  # Default month for the slider
    'Accident_Duration': 5  # Default duration in min (5 minutes)
}


def transform_features(raw):
    """
    Transform the raw feature dictionary into the format expected by the model.
    :param raw: Dictionary containing the raw features
    raw looks like:
    {
        'Traffic_Signal_Flag': user_input_value,  # e.g., 0 or 1
        'Crossing_Flag': user_input_value,  # e.g., 0 or 1
        'Highway_Flag': user_input_value,  # e.g., 0 or 1
        'Distance(mi)': user_input_value,  # e.g., 1.0 
        'Start_Hour': user_input_value, # e.g., 12 (hour of the day)
        'Start_Month': user_input_value, # e.g., 6 (month of the year)
        'Accident_Duration': user_input_value  # e.g., 5 (duration in minutes)
    }
    :return: Dictionary with transformed features suitable for the model
    returned dictionary will look like:
    {
        'Traffic_Signal_Flag': user_input_value,
        'Crossing_Flag': user_input_value,
        'Highway_Flag': user_input_value,
        'Distance(mi)': user_input_value,
        'Start_Hour_Sin': calculated_sin_value,
        'Start_Hour_Cos': calculated_cos_value,
        'Start_Month_Sin': calculated_sin_value,
        'Start_Month_Cos': calculated_cos_value,
        'Accident_Duration': normalized_duration_value
    }
    """
    # TODO


def predict_severity():
    # Transform the features to the format expected by the model
    transformed = transform_features(features)
    x = torch.tensor([list(transformed.values())], dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item() + 1
        return pred

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(11, 6))
ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())

# Style the map
# TODO
"""
Copy and paste the map styling code from your previous example here
"""

# Initial title
ax.set_title("Click on a location to predict severity", fontsize=14, pad=20)

red_dot = None

def on_click(event):
    global red_dot
    if event.inaxes == ax:
        if red_dot:
            red_dot.remove()
        red_dot = ax.plot(event.xdata, event.ydata, 'ro', markersize=6, transform=ccrs.Geodetic())[0]
        severity = predict_severity()
        ax.set_title(f"Clicked: ({event.xdata:.2f}, {event.ydata:.2f}) | Predicted Severity: {severity}", fontsize=14, pad=20)
        fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click)

sliders = []
slider_names = list(features.keys())

for i, name in enumerate(slider_names):
    # # Create an axes area for the slider at [left, bottom, width, height] within the figure
    ax_slider = plt.axes([0.15, 0.02 + i*0.03, 0.65, 0.02])

    if 'Flag' in name:
        slider = #TODO Allow binary values for flags (0 or 1)

    elif name == 'Start_Hour':
        slider = #TODO Allow hour from 0 to 23

    elif name == 'Start_Month':
        slider = #TODO Allow month from 1 to 12

    elif name == 'Distance(mi)':
        slider = #TODO Allow distance from 1.0 to 10 

    elif name == 'Accident_Duration':
        slider = #TODO Allow duration from 0 to 180 minutes

    else:
        slider = Slider(ax_slider, name, -1, 1, valinit=features[name])

    sliders.append(slider)

def update(val):
    for i, name in enumerate(slider_names):
        features[name] = sliders[i].val

    if red_dot:
        severity = predict_severity()
        ax.set_title(f"Predicted Severity: {severity}", fontsize=14, pad=20)
        fig.canvas.draw_idle()

for s in sliders:
    s.on_changed(update)

plt.subplots_adjust(top=0.88, bottom=0.35)  # Adjust to give room for the title
plt.show()
