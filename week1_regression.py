# TODO: Add import statements
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib

def main():
    print("Hello World")

# Assign the dataframe to this variable.
# TODO: Load the data
    bmi_life_data = pd.read_csv('bmi_life_expectancy.csv')
    print len(bmi_life_data)
    print bmi_life_data.info()

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
    bmi_life_model = LinearRegression()
    bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])
    

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
    laos_life_exp = bmi_life_model.predict(21.07931)
    
    print(laos_life_exp)





if __name__ == "__main__":
   main()
