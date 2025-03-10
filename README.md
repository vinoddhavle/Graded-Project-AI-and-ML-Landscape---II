# Graded-Project-AI-and-ML-Landscape---II
# Bike Sharing Demand Prediction
> A linear regression model to predict bike rental demand for BoomBikes, aiding post-pandemic business strategy.

## Table of Contents
* [General Information](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
- **Background**: BoomBikes, a US bike-sharing provider, aims to recover from COVID-19 losses by understanding factors influencing bike demand.
- **Business Problem**: Predict bike rental demand to optimize inventory, pricing, and marketing strategies post-lockdown.
- **Dataset**: 730-day bike-sharing data (2018-2019) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset), including:
  - Temporal features: season, month, holiday, weekday
  - Weather features: temperature, humidity, windspeed, weather situation
  - Target variable: `cnt` (total rentals per day)

## Conclusions
1. **Key Demand Drivers**: 
   - Yearly growth (`yr`): 2019 demand increased by 40% over 2018
   - Temperature (`temp`): 10°C increase → 2,300 more daily rentals
   - Clear weather (`weathersit_Clear`) boosts demand by 35% vs. rainy days

2. **Seasonality**: 
   - Peak demand in fall (+28% vs annual average)
   - Winter sees lowest demand (-22%)

3. **Multicollinearity Handling**:
   - Removed `atemp` (r = 0.99 with `temp`) and `workingday` (redundant with `holiday`)

4. **Model Performance**:
   - Achieved **R² = 0.84** on test data, indicating strong predictive power

## Technologies Used
- Python 3.10
- pandas 1.5.0 - Data manipulation
- scikit-learn 1.2.0 - Model building
- statsmodels 0.13.5 - VIF analysis
- matplotlib 3.7.0 - Visualization

## Acknowledgements
- Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu)
- Inspired by: UpGrad ML/AI curriculum
- Technical references: 
  - https://www.statsmodels.org
  - https://www.anaconda.com/
)
## Contact
Created by [) - Reach out at vinod.dhavle@gmail.com for collaborations!
