# Employee Attrition Prediction System

A full-stack web application for predicting employee attrition using a CatBoost machine learning model. The system consists of a React frontend and a Flask backend API.

## Project Structure

```
Assignment_Code/
├── V2_ML_Assignment.ipynb          # Jupyter notebook with model training
├── backend/
│   ├── app.py                       # Flask API server
│   ├── requirements.txt             # Python dependencies
│   ├── save_model.py                # Script to train and save model
│   └── model/                       # Directory for saved model (created after training)
│       └── attrition_model.cbm     # Trained CatBoost model
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js                   # Main React component
│   │   ├── App.css                  # Styling
│   │   ├── index.js                 # React entry point
│   │   └── index.css                # Global styles
│   └── package.json                 # Node.js dependencies
└── README.md                        # This file
```

## Features

- **Interactive Web Form**: User-friendly interface to input employee data
- **Real-time Predictions**: Get instant attrition predictions with confidence scores
- **Probability Visualization**: Visual representation of stay/leave probabilities
- **Feature Importance**: Display which factors most influence the prediction
- **Modern UI**: Beautiful, responsive design with gradient styling

## Setup Instructions

### Prerequisites

- Python 3.8+ with pip
- Node.js 14+ and npm
- The CSV data file used for training

### Step 1: Train and Save the Model

You have two options:

#### Option A: Using the Notebook (Recommended)
1. Open `V2_ML_Assignment.ipynb` in Jupyter
2. Run all cells to train the model
3. The last cell will save the model to `backend/model/attrition_model.cbm`

#### Option B: Using the Python Script
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Update the CSV filename in `save_model.py` if needed (currently expects the CSV in parent directory)
3. Run the script:
   ```bash
   python save_model.py
   ```

**Note**: Make sure the CSV file path in the script matches your actual data file location.

### Step 2: Set Up the Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask server:
   ```bash
   python app.py
   ```

   The API will be available at `http://localhost:5000`

### Step 3: Set Up the Frontend

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

   The frontend will open at `http://localhost:3000`

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Fill in the form with employee information:
   - **Department**: Employee's primary department
   - **Overtime**: Average monthly overtime hours
   - **Promotion Gap**: Years since last promotion/title change
   - **Job Satisfaction**: Current satisfaction level
   - **AI/Automation Risk**: Perceived risk level
   - **Recent Layoffs**: Whether department had layoffs in last 12 months
   - **Job Security**: Perceived job security level
   - **Market Demand**: Ease of finding similar role elsewhere
3. Click "Predict Attrition" to get the prediction
4. View the results showing:
   - Prediction (Stay/Leave)
   - Confidence percentage
   - Probability breakdown
   - Feature importance

## API Endpoints

### `GET /api/health`
Health check endpoint to verify the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /api/predict`
Predict employee attrition based on input features.

**Request Body:**
```json
{
  "Department": "Engineering",
  "Overtime": "11-20 hours",
  "Promotion_Gap": 2.5,
  "Job_Satisfaction": "Satisfied",
  "AI_Automation_Risk": "Medium",
  "Recent_Layoffs": "No",
  "Job_Security": "Secure",
  "Market_Demand": "Easy"
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "No (Likely to Stay)",
  "probability": {
    "stay": 0.65,
    "leave": 0.35
  },
  "confidence": 0.65,
  "feature_importance": {
    "Department": 12.5,
    "Overtime": 8.3,
    ...
  }
}
```

### `GET /api/features`
Get information about required features and their possible values.

## Model Details

- **Algorithm**: CatBoost Classifier
- **Target Variable**: Attrition (Binary: Yes/No)
- **Features**: 8 features including categorical and numerical
- **Evaluation Metric**: F1-Score (optimized for imbalanced data)

## Troubleshooting

### Model Not Found Error
- Ensure you've trained and saved the model first (Step 1)
- Check that `backend/model/attrition_model.cbm` exists
- Verify the model path in `backend/app.py` matches your file location

### CORS Errors
- Make sure Flask-CORS is installed: `pip install flask-cors`
- Verify the backend is running on port 5000
- Check that the frontend proxy is configured in `package.json`

### Port Already in Use
- Backend: Change port in `backend/app.py` (default: 5000)
- Frontend: React will prompt to use a different port automatically

### CSV File Not Found
- Update the CSV file path in the notebook or `save_model.py`
- Ensure the CSV file is in the correct location

## Development

### Backend Development
- The Flask app runs in debug mode by default
- API logs will show in the terminal
- Model is loaded once on startup

### Frontend Development
- React hot-reload is enabled
- Changes to `src/` files will automatically refresh
- Check browser console for any errors

## Production Deployment

For production deployment:

1. **Backend**:
   - Set `debug=False` in `app.py`
   - Use a production WSGI server (e.g., Gunicorn)
   - Set up proper environment variables
   - Configure CORS for your domain

2. **Frontend**:
   - Build the production bundle: `npm run build`
   - Serve the `build/` directory with a web server (e.g., Nginx)
   - Update API URL in environment variables

## License

This project is for educational purposes as part of the IT5514 Applied Machine Learning course.

## Author

Created for the Employee Attrition Prediction Assignment.


