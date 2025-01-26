# MDST Traffic Accident Prediction

A **machine learning model** trained on real-world traffic data to predict accident severity.

## Features
- **Inputs:** Latitude, Longitude, Weather, Time, Traffic Conditions...
- **Output:** Predicted Accident Severity (from 1 to 4).
- Implemented in Python with a Neural Network using PyTorch.

## Deliverables
1. **Cleaned and Preprocessed Dataset:** Combines accident and traffic flow data.
2. **Visualizations:** Explories relationships between accident severity and key features.
3. **Trained Neural Network Model:** Capable of predicting accident severity.
4. **Interactive Web Application:**
   - A map-based interface for location selection.
   - Input fields for relevant feature variables.
   - Real-time accident severity predictions displayed to users.

## Project Agenda
| Week  | Date   | Agenda                          |
|-------|--------|---------------------------------|
| 1     | 1/26   | Icebreakers + Environment Setup |
| 2     | 2/2    | Exploratory Data Analysis (EDA) |
| 3     | 2/9    | Neural Nets                     |
| 4     | 2/16   | PyTorch                         |
| 5     | 2/23   | Basic NN Model                  |
| -     | -      | Spring Break                    |
| -     | -      | Spring Break                    |
| 6     | 3/16   | Model Development               |
| 7     | 3/23   | Model Development               |
| 8     | 3/30   | Interactive Map                 |
| 9     | 4/16   | Interactive Map                 |
| 10    | 4/23   | Final Expo Prep                 |

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Download Datasets
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

https://archive.ics.uci.edu/dataset/608/traffic+flow+forecasting

### 3. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows
```

### 4. Install Dependencies
Install the required Python packages:
```bash
pip3 install -r requirements.txt
```

### 5. Verify Installation
Run the following to ensure the environment is set up correctly:
```bash
python3 -m pip list
```

## Running the Project
1. **Data Preprocessing:** Execute the preprocessing script to clean and combine datasets.
```bash
```

2. **Training the Model:** Train the neural network model using PyTorch.
```bash
```

3. **Visualizations:** Generate and explore visual insights.
```bash
```

4. **Web Application:** Launch the interactive app.
```bash
```

## Requirements
The required Python packages are listed in `requirements.txt`:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
torch
torchvision
```

### Note:
For PyTorch and torchvision, ensure you install the versions suitable for your hardware (e.g., CPU or GPU with CUDA). Refer to the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

