import joblib
import pandas as pd

def predict_flowers():
    print('=' * 50)
    print('Flower Classification System')
    print('=' * 50)
    sepal_length = float(input(f'Sepal Length: '))
    sepal_width = float(input(f'Sepal Width: '))
    petal_length = float(input(f'Petal Length: '))
    petal_width = float(input(f'Petal Width: '))
    
    model = joblib.load(r'C:\Users\pc\Documents\MACHINE LEARNING\Syntecxhub-Internship-Programme\Flower Classification\datasets\decision tree model.pkl')

    user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    prediction = model.predict(user_input)[0]

    if isinstance(prediction, str):
        result = prediction 

    else: 
        result = prediction

    print('=' * 50)
    print(f'Predicted Specie is: {result}')
    print('=' * 50)



if __name__ == "__main__":
    predict_flowers()