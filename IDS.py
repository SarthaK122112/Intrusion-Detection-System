import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and preprocess the data with proper encoding
try:
    df = pd.read_csv('philips.csv', encoding='latin1', on_bad_lines='skip')
except Exception as e:
    print(f"Error loading file: {e}")
    exit()




# Feature engineering
# 1. Protocol encoding
protocol_encoder = LabelEncoder()
df['Protocol_encoded'] = protocol_encoder.fit_transform(df['Protocol'])

# 2. Packet length features - handle potential non-numeric values
df['Length'] = pd.to_numeric(df['Length'], errors='coerce').fillna(0).astype(int)


# 3. Time delta features
df['Time'] = pd.to_numeric(df['Time'])
df['Time_delta'] = df['Time'].diff().fillna(0)

# 4. Source/Destination features (simplified)
df['Source_prefix'] = df['Source'].apply(lambda x: x.split('.')[0] if '.' in x else 'other')
df['Dest_prefix'] = df['Destination'].apply(lambda x: x.split('.')[0] if '.' in x else 'other')

source_encoder = LabelEncoder()
dest_encoder = LabelEncoder()
df['Source_encoded'] = source_encoder.fit_transform(df['Source_prefix'])
df['Dest_encoded'] = dest_encoder.fit_transform(df['Dest_prefix'])

# 5. Behavioral features (simplified example)
df['Is_MDNS'] = df['Protocol'].apply(lambda x: 1 if 'MDNS' in x else 0)
df['Is_DHCP'] = df['Protocol'].apply(lambda x: 1 if 'DHCP' in x else 0)
df['Is_NTP'] = df['Protocol'].apply(lambda x: 1 if 'NTP' in x else 0)

# For this example, we'll simulate labels since the dataset doesn't have them
# In a real scenario, you would have labeled data indicating normal vs malicious traffic
np.random.seed(42)
df['Label'] = np.random.randint(0, 2, size=len(df))  # 0 = normal, 1 = malicious

# Select features
features = ['Protocol_encoded', 'Length', 'Time_delta', 
            'Source_encoded', 'Dest_encoded', 
            'Is_MDNS', 'Is_DHCP', 'Is_NTP']
X = df[features]
y = df['Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

# Print best model
best_model = max(results, key=results.get)
print(f"Best performing model: {best_model} with accuracy {results[best_model]:.4f}")