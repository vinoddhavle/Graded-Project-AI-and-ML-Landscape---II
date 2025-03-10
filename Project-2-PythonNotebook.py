# Bike Sharing Demand Prediction - Final Solution
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load and prepare data
df = pd.read_csv('Data.csv')

# Remove non-predictive columns
df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1, inplace=True)

# Split data first to prevent leakage
X = df.drop('cnt', axis=1)
y = df['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom categorical mapping
def convert_categorical(df_split):
    # Convert numeric categories to meaningful labels
    season_map = {1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'}
    weather_map = {1: 'clear', 2: 'mist', 3: 'light_precip', 4: 'heavy_precip'}
    
    df_split = df_split.copy()
    df_split['season'] = df_split['season'].map(season_map)
    df_split['weathersit'] = df_split['weathersit'].map(weather_map)
    df_split['mnth'] = df_split['mnth'].astype(str)
    df_split['weekday'] = df_split['weekday'].astype(str)
    
    # Remove problematic column
    if 'workingday' in df_split.columns:
        df_split.drop('workingday', axis=1, inplace=True)
    
    return df_split

# Apply categorical conversions
X_train = convert_categorical(X_train)
X_test = convert_categorical(X_test)

# Create dummy variables with proper handling
def create_dummies(df_split, reference_cols=None):
    categorical_cols = ['season', 'weathersit', 'mnth', 'weekday']
    dummies = pd.get_dummies(df_split, columns=categorical_cols, drop_first=True)
    
    # Ensure numeric types
    dummies = dummies.astype(float)
    
    # Align with reference columns if provided
    if reference_cols is not None:
        dummies = dummies.reindex(columns=reference_cols, fill_value=0)
    
    return dummies

# Create dummies using training set as reference
X_train_dummies = create_dummies(X_train)
X_test_dummies = create_dummies(X_test, reference_cols=X_train_dummies.columns)

# VIF analysis function
def calculate_vif(dataframe):
    vif_data = add_constant(dataframe)
    vif = pd.DataFrame()
    vif["Feature"] = vif_data.columns
    vif["VIF"] = [variance_inflation_factor(vif_data.values, i) 
                  for i in range(vif_data.shape[1])]
    return vif.sort_values('VIF', ascending=False)

# Remove constant column if present
if 'const' in X_train_dummies.columns:
    X_train_dummies = X_train_dummies.drop('const', axis=1)
    X_test_dummies = X_test_dummies.drop('const', axis=1)

# Perform VIF analysis
print("=== Initial VIF Analysis ===")
initial_vif = calculate_vif(X_train_dummies)
print(initial_vif.head(10))

# Remove high-VIF features iteratively
high_vif_features = initial_vif[initial_vif['VIF'] > 10]['Feature'].tolist()
if 'const' in high_vif_features:
    high_vif_features.remove('const')

while high_vif_features:
    feature_to_remove = high_vif_features[0]
    X_train_dummies = X_train_dummies.drop(feature_to_remove, axis=1)
    X_test_dummies = X_test_dummies.drop(feature_to_remove, axis=1, errors='ignore')
    
    print(f"\nRemoved: {feature_to_remove}")
    current_vif = calculate_vif(X_train_dummies)
    high_vif_features = current_vif[current_vif['VIF'] > 10]['Feature'].tolist()
    
    if 'const' in high_vif_features:
        high_vif_features.remove('const')

print("\n=== Final VIF Analysis ===")
print(calculate_vif(X_train_dummies).head(10))

# Train final model
model = LinearRegression()
model.fit(X_train_dummies, y_train)

# Evaluate performance
y_pred = model.predict(X_test_dummies)
r2 = r2_score(y_test, y_pred)

print("\n=== Final Model Performance ===")
print(f"Adjusted R² Score: {r2:.4f}")
print(f"Features Used ({len(X_train_dummies.columns)}):")
print(list(X_train_dummies.columns))