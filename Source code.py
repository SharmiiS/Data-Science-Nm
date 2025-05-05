# 1. Upload and Load Dataset
from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv("/content/surveyyy.csv")

# 2. Explore the Data
print(df.shape)
print(df.columns)
df.info()
df.describe()

# 3. Preprocessing
# Drop unnecessary columns and handle missing values
# Check if 'comments' and 'state' columns exist before dropping
if 'comments' in df.columns and 'state' in df.columns:
    df = df.drop(['comments', 'state'], axis=1)
else:
    print("Warning: 'comments' or 'state' column not found in the DataFrame.")

df = df.dropna()

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Feature Scaling and Split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df_encoded.drop('treatment_Yes', axis=1)
y = df_encoded['treatment_Yes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Build Gradio App
!pip install gradio
import gradio as gr

def predict_mental_health(*inputs):
    input_df = pd.DataFrame([inputs], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return "Needs Treatment" if prediction[0] == 1 else "No Treatment Needed"

inputs = [gr.Number(label=col) for col in X.columns]
output = gr.Text(label="Prediction")

gr.Interface(fn=predict_mental_health, inputs=inputs, outputs=output,
             title="🧠 Mental Health Risk Predictor",
             description="Enter the details to predict whether a person needs mental health treatment.").launch()
