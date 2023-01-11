from cmath import isnan
import pandas as pd
from numpy import dot
import numpy as np
from numpy.linalg import norm

# Reading the train data
trainFrame = pd.read_csv("netflix/TrainingRatings.txt", names = ["MovieID","CustomerID","Rating"])
#trainFrame = trainFrame.sort_values(by='CustomerID')

# Calculating Mean of user ratings 
ViBar = trainFrame.groupby(by="CustomerID", as_index = False)['Rating'].mean()
trainMeanArray = np.array(ViBar['Rating'])[:, np.newaxis]

# Create a matrix with customers as rows and movies as columns
trainDataFrame = trainFrame.pivot(index = 'CustomerID', columns = 'MovieID', values = 'Rating')
RatingDiffTrainFrame = trainDataFrame - trainMeanArray

#If a user has not rated for a movie, then his rating for that movie is zero and Vij-ViBar should also be zero
trainDataFrame.fillna(0, inplace=True)
RatingDiffTrainFrame.fillna(0, inplace=True)
# Converts to a 2D matrix, with all ratings
trainMatrix = trainDataFrame.to_numpy()
RatingDiffTrainMatrix = RatingDiffTrainFrame.to_numpy()

# Reading the test data
testFrame = pd.read_csv("netflix/TestingRatings.txt", names = ["MovieID","CustomerID","Rating"])
# Create a matrix with customers as rows and movies as columns
testDataFrame = testFrame.pivot(index = 'CustomerID', columns = 'MovieID', values = 'Rating')

# Computing Similarity measure
RatingsDiffTrainNorm = norm(RatingDiffTrainMatrix, axis=1)[:, np.newaxis]
ProductofRatingsDiffNorm = RatingsDiffTrainNorm * RatingsDiffTrainNorm

dotProduct = dot(RatingDiffTrainMatrix, RatingDiffTrainMatrix.T)
collinearity = dotProduct / ProductofRatingsDiffNorm

# Predicting test user ratings
trainMovies = trainDataFrame.columns.values
trainCustomers = trainDataFrame.index.values

predictedRatings = []
actualRatings = []

# Iterate over rows of test file, get the customer id and movie id. 
for index, row in testFrame.iterrows():
   testCustomerID = row['CustomerID']
   testMovieID = row['MovieID']
   
   actualRating = row['Rating']
   actualRatings.append(actualRating)
   
   avgRating = ViBar.query('CustomerID == @testCustomerID')['Rating'].values

   customerSimilarity = collinearity[np.where(trainCustomers == testCustomerID)]
   
   ratingDiff = RatingDiffTrainMatrix[:, (np.where(trainMovies == testMovieID)[0][0])]
   
   K = np.abs(customerSimilarity).sum(axis=1)
   weightedAvg = (np.dot(customerSimilarity, ratingDiff.T)/K)
   
   predictedRating = np.round(avgRating + weightedAvg)[0]
   predictedRatings.append(predictedRating)
   
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from math import sqrt

predictedRatings = [0 if pd.isna(x) else x for x in predictedRatings]
mse = mean_squared_error(predictedRatings, actualRatings)
rmse = sqrt(mse)

print("Mean Squared Error: ", mse)
print("Root Mean Squared Error:", rmse)

#accuracy = accuracy_score(actualRatings, predictedRatings)
#print(accuracy)
