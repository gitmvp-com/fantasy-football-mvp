# Fantasy Football Prediction MVP

A simplified LSTM-based prediction system for forecasting fantasy football points for wide receivers.

## Features

- **LSTM Neural Network**: Predicts next season's fantasy points per game for wide receivers
- **Feature Engineering**: Rolling averages, per-game statistics, and season flags
- **Model Training**: Custom weighted MSE loss for better prediction accuracy
- **Evaluation**: Performance metrics (RMSE, MAE, R²) and visualization

## Project Structure

```
├── data/                      # Sample data directory
│   └── sample_data.xlsx      # Example fantasy football data
├── models/                    # Trained model storage
├── results/                   # Prediction outputs and plots
├── scripts/
│   ├── model.py              # LSTM model architecture
│   ├── training.py           # Training logic with custom loss
│   ├── prediction.py         # Prediction functions
│   └── feature_engineering.py # Data preprocessing
├── main.py                    # Main execution script
└── requirements.txt          # Python dependencies
```

## Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn
- openpyxl
- matplotlib
- seaborn

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Data

Place your fantasy football data in `data/cleaned_fantasy_football_data.xlsx` with these columns:
- `Player`, `Year`, `Age`, `G`, `Tgt`, `Rec`, `RecYds`, `RecTD`, `TD`, `FantPt`, `FantPtHalf`, `FantPos`

### 2. Run the Model

```bash
python main.py
```

This will:
- Load and preprocess the data
- Engineer features (rolling averages, per-game stats)
- Train the LSTM model
- Evaluate performance
- Generate predictions for the next season
- Save results to `results/predictions_2025.xlsx`

### 3. View Results

Check the `results/` directory for:
- `predictions_2025.xlsx` - Predicted fantasy points for next season
- `actual_vs_predicted.png` - Scatter plot of predictions vs actual values
- `distribution_curve.png` - Distribution comparison
- `correlation_matrix.png` - Feature correlation heatmap

## Model Architecture

- **Input**: Player statistics and engineered features
- **LSTM Layers**: 2 layers with 128 hidden dimensions
- **Dropout**: 0.2 for regularization
- **FC Layers**: 128 → 64 → 32 → 1
- **Loss Function**: Weighted MSE (higher weight for elite players)

## Key Features

### Feature Engineering
- Per-game statistics (TD/G, RecYds/G, FantPt/G, Tgt/G)
- Rolling averages (2-year and 3-year)
- Season flags (rookie, second year, veteran)
- Target shifting (predict next year's performance)

### Custom Loss Function
- Higher weight for elite players (>15 pts/game): 4x
- Lower weight for low performers (<4 pts/game): 3x
- Normal weight for mid-tier players: 1x

## Sample Data Format

Create a sample dataset with this structure:

| Player | Year | Age | G | Tgt | Rec | RecYds | RecTD | TD | FantPt | FantPtHalf | FantPos |
|--------|------|-----|---|-----|-----|--------|-------|----|---------|-----------|---------|
| Player1| 2023 | 25  | 16| 120 | 80  | 1200   | 8     | 8  | 180     | 200       | WR      |

## MVP Simplifications

- No authentication required
- Uses local file storage (no database)
- Simplified data format (Excel only)
- Pre-configured hyperparameters
- Single position focus (wide receivers only)

## Future Enhancements

- Support for multiple positions (RB, TE, QB)
- Web interface for predictions
- Real-time data fetching from fantasy APIs
- Hyperparameter tuning interface
- Model versioning and comparison

## License

MIT License