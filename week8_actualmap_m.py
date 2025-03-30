import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
        # TODO: Implement the forward pass using fc1, dropout
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = AccidentSeverityModel()
model.load_state_dict(torch.load('best_accident_severity_model.pth'))
model.eval()

# Use default features for every prediction
features = {
    'Traffic_Signal_Flag': 0,
    'Crossing_Flag': 0,
    'Highway_Flag': 1,
    'Distance(mi)': -0.3,
    'Start_Hour_Sin': 0.0,
    'Start_Hour_Cos': 1.0,
    'Start_Month_Sin': 0.0,
    'Start_Month_Cos': 1.0,
    'Accident_Duration': -0.02
}

def predict_severity():
    # TODO: Create a tensor from the feature values, pass it through the model, and return the predicted severity (add 1 to the output class index)

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(11, 6))
ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())

# TODO: Copy and paste what you did for week8_map_m.py 
# Add map features (land, ocean, lakes, borders, states, coastline)


ax.set_title("Click on a location to predict severity", fontsize=14, pad=20)

red_dot = None

def on_click(event):
    # TODO: When user clicks on the map, show a red dot and update the title with predicted severity at that location

fig.canvas.mpl_connect('button_press_event', on_click)

plt.subplots_adjust(top=0.88)
plt.show()
