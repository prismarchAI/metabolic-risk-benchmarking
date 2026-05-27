import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

#load the dataset
try:
    df = pd.read_csv('real_diabetes_data.csv')
except FileNotFoundError:
    print("Error: 'real_diabetes_data.csv' not found.")
    exit()
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Applying Feauture Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#training 100 decision trees together
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled,y_train)

#evaluating the new engine
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test,predictions)

print("==================================================")
print("      UPGRADED ENSEMBLE DIAGNOSTIC REPORT        ")
print("==================================================")
print(f"Dataset Size Analyzed: {len(df)} patient records")
print(f"New Random Forest Accuracy: {accuracy * 100:.2f}%\n")
print("Clinical Metric Breakdown Matrix:")
print(classification_report(y_test, predictions, target_names=['Tested Negative (0)', 'Tested Positive (1)']))
print("==================================================")