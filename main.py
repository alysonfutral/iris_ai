# intro to classification

# so far we have learned about predictions using regression with numbers
    # now we will learn about classification

# What we mean by this as instead of predicting data with numbers, we will predict data with categories

# examples
# What if you wanted your model to predict if an animal is a dog or a cat depending on paw size, fur pattern, and weight? What if you wanted your model to predict if a new email was spam or not depending on the contents of the email?

# Classification is a machine learning technique where a model will learn how to assign a provided class label to provided data. The important part is to keep in mind is that the classes are pre-defined by the user.

# A linear classifier is a type of classifier that "splits" a data set into two groups.

# examples from reading
  # Using the data, we want to train a linear classifier that labels a           flower as a rose or a daisy depending on its color.
  #Conceptually, we want our trained classifier to do something like this,       where it can split flowers apart from one another

# ISSUES THAT CAN HAPPEN
# Say we want to predict a white rose. We run into an issue, since our model only uses flower color to determine what kind of flower that will be predicted. Since all of the daisies are white, it might wrongly predict a white rose as a daisy.
# TO FIX THE ISSUE
#However, this can be solved by adding more features to our model to make it more adaptable.

# LETS USE SUPPORT VECTOR MACHINE TO CLASSIFY CATAGORIES (SVM)

#What SVM does to find the best line is draw a line (decision boundary) between the most innermost data points from opposing classes. The two most innermost data points would be the two that have been circled in black. (image in notes)


#The first line will add the svm library from sklearn so that we can use the svm classification model to train our data. The second line will add the pandas library so that we can use pandas dataframes. pandas is a library used by data scientists that allows for easy data manipulation.
from sklearn import svm
import pandas as pd


# before we train our model, we have to encode the species data into numerical values. The species data is currently stored as a string value. However, the machine learning model needs the data to be a numerical value. To convert, we will use a LabelEncoder
from sklearn.preprocessing import LabelEncoder

#Now let's take the data from the Iris.csv file and add it to a pandas dataframe.
df = pd.read_csv("Iris.csv")

print(df)

print()

#############################STEP 2
#Let's take the "SepalLength", "SepalWidth", "PetalLength", and "PetalWidth" and separate them into a smaller data frame so that we can use them to train our model.
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
Y = df[['Species']]

print(X)
print(Y)

print()
#############################STEP 3
#Now we will use the LabelEncoder to encode the data so that the model can understand the data.
le = LabelEncoder()

yEncoded = le.fit_transform(Y['Species'])
#************Note that we passed in "Species" instead of just Y. This is because the le.fit_transform function is expecting a list and not the whole dataframe. Y['Species'] will return the values underneath the species column.
print(yEncoded)

print()

#############################STEP 4
#The original text data for species has been encoded as numerical data! Now lets train our model.
IrisPredictionModel = svm.SVC()

IrisPredictionModel.fit(X, yEncoded)

# We are trying to predict a flower with SepalLengthCm of 4.9, SepalWidthCm of 3, PetalLengthCm of 1.4, and a PetalWidthCm of 0.2. After printing the prediction, we receive this value as the output.
prediction = IrisPredictionModel.predict([[4.9, 3, 1.4, .2]])

print(prediction)

print()

#############################STEP 5
#our final step will be to translate the encoded number back to its corresponding value.
returnToOriginal = le.inverse_transform(prediction)

print(returnToOriginal)

print()
print()

#######################################################################################################

# time to use predicting algortihm to predict the species of the Iris flower
from sklearn import svm 
import pandas as pd
from sklearn.preprocessing import LabelEncoder


print("Welcome to the Iris Flower Prediction Program! \n")
print("We will use Support Vector Machines to predict what kind of iris flower you have! \n")
print("All you need to do is supply some information about the flower! \n") #

while True:
  print("1. Make a prediction")
  print("2. Exit the program")

  choice = input("")
  if choice == "1":
    sepalLength = float(input("What is the Sepal Length in cm? \n"))
    sepalWidth = float(input("What is the Sepal Width in cm? \n"))
    petalLength = float(input("What is the Petal Length in cm? \n"))
    petalWidth = float(input("What is Petal Width in cm? \n"))

    df = pd.read_csv("Iris.csv")
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    Y = df[['Species']]

    le = LabelEncoder()
    yEncoded = le.fit_transform(Y['Species'])

    irisPredictionModel = svm.SVC()
    irisPredictionModel.fit(X, yEncoded)

    prediction = irisPredictionModel.predict([[ sepalLength,  sepalWidth ,  petalLength, petalWidth]])

    returnToOriginal = le.inverse_transform(prediction)
    print("The type of iris flower is " + returnToOriginal[0])
  elif choice == "2":
    break

print("Goodbye!")