import joblib
import pandas as pd

def predict_new_flower():
    print("=" * 50)
    print("🌸 IRIS SPECIES PREDICTOR")
    print("=" * 50)

    print("\nEnter flower measurements (in cm):")
    sepal_length = float(input("Sepal Length: "))
    sepal_width = float(input("Sepal Width: "))
    petal_length = float(input("Petal Length: "))
    petal_width = float(input("Petal Width: "))

    model_path = r'C:\Users\pc\Documents\MACHINE LEARNING\Syntecxhub-Internship-Programme\Flower Classification\datasets\decision tree model.pkl'
    pipe1 = joblib.load(model_path)

    user_input = pd.DataFrame[[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = pipe1.predict(user_input)[0] 
    
    if isinstance(prediction, str):
        result = prediction
    else:
        species_names = ['Setosa', 'Versicolor', 'Virginica']
        result = species_names[int(prediction)]

    print(f" Predicted Species: {result}")
    print("=" * 50)

if __name__ == "__main__":
    predict_new_flower()