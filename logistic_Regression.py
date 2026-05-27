import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


try:
    df = pd.read_csv('real_diabetes_data.csv')
except FileNotFoundError:
    print("Error: 'real_diabetes_data.csv' not found.")
    exit()

X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','BMI','DiabetesPedigreeFunction','Age']]
y=df['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model =  LogisticRegression(max_iter = 200)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy =  accuracy_score(y_test, predictions)

print("==================================================")
print("       REAL-WORLD DIABETES DIAGNOSTIC REPORT      ")
print("==================================================")
print(f"Dataset Size Analyzed: {len(df)} patient Records")
print(f"Engine Classification Accuracy: {accuracy *100:.2f}%\n")
print("Clinical Metric Breakdown Matrix:")
print(classification_report(y_test,predictions,target_names=['Tested Negative (0)','Tested Positive (1)']))
print("==================================================")

print("\n==================================================")
print("          RUNNING LIVE PATIENT DIAGNOSTIC         ")
print("==================================================")

new_patient_data = {
    'Pregnancies': [2],               # 2 previous pregnancies
    'Glucose': [148],                 # High blood sugar level
    'BloodPressure': [72],             # Normal diastolic pressure
    'SkinThickness': [35],             # Triceps skin fold thickness (mm)
    'BMI': [33.6],                    # Obese weight class
    'DiabetesPedigreeFunction': [0.627], # High genetic predisposition factor
    'Age': [50]                       # 50 years old
}

new_patient_df = pd.DataFrame(new_patient_data)
live_prediction = model.predict(new_patient_df)

live_probability = model.predict_proba(new_patient_df)[0][1]

print(f"Calculated Diabetes Risk Probability: {live_probability * 100:.2f}%")

if live_prediction[0] == 1:
    print("RISK DETECTED")
else:
    print("NO RISK DETECTED")

print("==================================================")
