# Deep_Learning_Challenge: Charity Funding Predictor

Objective and Summary:

The purpose of this analyis is that with the help of deep learning skills (machine learning and neural networks) to predict which applicants will be sucessful if they get funded by the Alphabet Soup Charity.
After processing the data, I compiled, trained and evaluated the model. Reached almost 73% of model accuracy. (Please see CharityFundingPredictor Jupyter Notebook file). 
Then tried to optimized the model, the first try, the accuracy didn't improve much (Please see the first optimizer file by dropping one more non-beneficial ID columns 'Special_Consierations' than EIN and NAME: AlphabetSoupCharity_Optimization.ipynb Jupyter notebook)
Then tried the second optimizer model (Jupyter Notebook 'ASC_Optimier2.ipynb) ended up with a boosted accuracy above 75%.
Please see the report and conclusion for more detail at the end of this readme file:


Background:

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.
From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

		EIN and NAME—Identification columns
		APPLICATION_TYPE—Alphabet Soup application type
		AFFILIATION—Affiliated sector of industry
		CLASSIFICATION—Government organization classification
		USE_CASE—Use case for funding
		ORGANIZATION—Organization type
		STATUS—Active status
		INCOME_AMT—Income classification
		SPECIAL_CONSIDERATIONS—Special consideration for application
		ASK_AMT—Funding amount requested
		IS_SUCCESSFUL—Was the money used effectively


Step 1: Preprocess the data

	Using the knowledge of Pandas and the Scikit-Learn’s StandardScaler(), we will need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Step 2
	Using the information we have provided in the starter code, follow the instructions to complete the preprocessing steps.

		Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in our dataset:
		Drop the EIN and NAME columns.
		Determine the number of unique values for each column.
		For those columns that have more than 10 unique values, determine the number of data points for each unique value.
		Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
		Use pd.get_dummies() to encode categorical variables

Step 2: Compile, Train, and Evaluate the Model
	
	Using our knowledge of TensorFlow, we will design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. We will need to think about how many inputs there are before determining the number of neurons and layers in our model. Once we have completed that step, we will compile, train, and evaluate our binary classification model to calculate the model’s loss and accuracy.

		Continue using the jupter notebook where we have already performed the preprocessing steps from Step 1.
		Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
		Create the first hidden layer and choose an appropriate activation function.
		If necessary, add a second hidden layer with an appropriate activation function.
		Create an output layer with an appropriate activation function.
		Check the structure of the model.
		Compile and train the model.
		Create a callback that saves the model's weights every 5 epochs.
		Evaluate the model using the test data to determine the loss and accuracy.
		Save and export our results to an HDF5 file, and name it AlphabetSoupCharity.h5.

Step 3: Optimize the Model

	Using our knowledge of TensorFlow, optimize our model in order to achieve a target predictive accuracy higher than 75%. If we can't achieve an accuracy higher than 75%, we will need to make at least three attempts to do so.
	Optimize our model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

		Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
		Dropping more or fewer columns.
		Creating more bins for rare occurrences in columns.
		Increasing or decreasing the number of values for each bin.
		Adding more neurons to a hidden layer.
		Adding more hidden layers.
		Using different activation functions for the hidden layers.
		Adding or reducing the number of epochs to the training regimen.


		Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.
		Import our dependencies, and read in the charity_data.csv to a Pandas DataFrame.
		Preprocess the dataset like we did in Step 1, taking into account any modifications to optimize the model.
		Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
		Save and export our results to an HDF5 file, and name it AlphabetSoupCharity_Optimization.h5.

	Please see the Optimization Jupyter Notebook 'ASC_Optimier2.ipynb & AlphabetSoupCharity_Optimization.ipynb'

Step 4: Write a Report on the Neural Network Model

Overview of the analysis:
	
	The purpose of this analyis is that with the help of deep learning skills (machine learning and neural networks) to predict which applicants will be sucessful if they get funded by the Alphabet Soup Charity.

Data Preprocessing
	
	What variable(s) are considered the target(s) for your model?
		MA Answer: IS_SUCCESSFUL comumn is the our model's target.
	What variable(s) are considered to be the features for your model?
		MA Answer: Almost all columns except the ones we dropped. So NAME, APPLICATION, TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT,SPECIAL_CONSIDERATIONS, STATUS, and ASK_AMT are going to be considered as the features of our model.
	What variable(s) are neither targets nor features, and should be removed from the input data?
		MA Answer: STATUS
	
Compiling, Training, and Evaluating the Model
	
	Were you able to achieve the target model performance?
		MA Asnwer: Yes, I was able to achieve the targer model to 78% which is above the 75% project's goal.
	What steps did you take to try and increase model performance?
		MA Answer: I did try first to drop extra column but found out the accuract decreases. So I had to convert the NAME col. into the data points similar to APPLICATION TYPE & CLASSIFICATIOn columns.

Summary:
	- MA Answer: When we increase the accuracy of our model to 78% so we should be able more accurately classify each of the points in our test data. 