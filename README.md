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

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
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
pandas==1.5.3
numpy==1.24.2
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
scipy==1.10.1
torch==1.13.1
torchvision==0.14.1
```

### Note:
For PyTorch and torchvision, ensure you install the versions suitable for your hardware (e.g., CPU or GPU with CUDA). Refer to the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

