# Reading the csv file from the directory specified below in setwd function.
setwd('/home/omar/Desktop/ML_Assign')
origData <- read.csv2('iris_sub_21655.csv', sep = ',', header = TRUE, stringsAsFactors = FALSE)

# Converting data type of petal and species columns from char to numeric.
origData$petal <- as.numeric(origData$petal)
origData$species <- as.numeric(origData$species)

# creating a new vector which represents the classes of plants as a number.
# 1 represents Setosa and -1 represents Versicolor. This vector is used later in
# covarianceAndCorrelationFunction to calculate the covariance and correlation.
speciesNumeric <- origData$X
levels(speciesNumeric) <- c(levels(speciesNumeric), 1, -1)
speciesNumeric[speciesNumeric=='setosa'] <- 1
speciesNumeric[speciesNumeric=='versicolor'] <- -1
speciesNumeric <- as.numeric(as.character(speciesNumeric))

# Initializing the global variables that are used later on in many functions through the code.
bias <- -1
biasWeight <- 0.05
sepalWeight <- -0.02
petalWeight <- 0.02
flag <- TRUE
iterationsCounter <- 0
learningRate <- 0.2

# These two global vectors are used to store the value of error along the learning process.
# actualErrorValues contains the real value of the error (h - t).
# thresholdErrorValues contains the value of the error after applying threshold to h (g - t).
actualErrorValues <- vector()
thresholdErrorValues <- vector()

# The training function is called to start the training process of the perceptron. It contains 
# a main while loop which keeps iterating as far as an error is detected in the training process.
# the while loop terminates only when a complete iteration is done with no error in the classification in sample data.
training <- function()
{
  while(flag){
    iterationsCounter <<- iterationsCounter + 1
    flag <<- FALSE
    # Iterate over the dataset provided
    for (count in 1:nrow(origData)) {
      sepal <- origData[count, 2]
      petal <- origData[count, 3]
      # calculating the summation of each weight multiplied by its corresponding weight. 
      h <- (bias * biasWeight) + (sepal * sepalWeight) + (petal * petalWeight)
      # applying thresholding to h value.
      g <- thresholdFunction(h)
      correctValue <- 0
      
      if (origData[count, 4] == 'setosa'){
        correctValue <- 1
        # if the correct class is Setosa but the algorithm classified it as Versicolor, then update 
        # the wights of the inputs
        if (g <= 0){ 
          flag <<- TRUE 
          updateWeights(g, 1, petal, sepal)
        }
      } else{
        # if the correct class is Versicolor but the algorithm classified it as Setosa, then update 
        # the wights of the inputs
        correctValue <- -1
        if (g > 0){ 
          flag <<- TRUE
          updateWeights(g, -1, petal, sepal)
        }
      }
      # Store the value of error for each input in the 2 global vectors mentioned above.
      actualErrorValues[(iterationsCounter - 1) * 100 + count] <<- h - correctValue
      thresholdErrorValues[(iterationsCounter - 1) * 100 + count] <<- g - correctValue
    }
  }
  print(paste("Training is finished successfully with learning rate = (", 
              learningRate, "). Number of iterations = ", iterationsCounter))
}

# This function applies threshold to the h value calculated in the training function above.
# Any value greater than zero is evaluated as 1 which stands for Setosa class.
# Any value less than zero is evaluated as -1 which stands for Versicolor class.
thresholdFunction <- function(hValue)
{
  if (hValue > 0){
    return (1)
  }
  return (-1)
}

# The updateWeights is called by the training function in case an error occured due to wrong classification. 
updateWeights <- function(gValue, tValue, petal, sepal)
{
  biasWeight <<- (biasWeight - (learningRate * (gValue - tValue) * bias))
  petalWeight <<- (petalWeight - (learningRate * (gValue - tValue) * petal))
  sepalWeight <<- (sepalWeight - (learningRate * (gValue - tValue) * sepal))
}

# This function is called after training the network in order to test its output.
# It takes 2 inputs. The 1st one is the sepal length of the plant and the 2nd one is its petal length.
# The output is a string that represents the class of the plant.
testing <- function(inSepal, inPetal)
{
  y = (bias * biasWeight) + (inSepal * sepalWeight) + (inPetal * petalWeight)
  if (y > 0){
    print ("Setosa")
  }else{
    print("Versicolor")
  }
}

# This function is used to plot the relationship between the different variables affecting the network.
# It takes one input that is used to select which relation to plot. The possible input values are demonstrated in
# below in the function itself.
plottingFunction <- function(selection)
{
  # scaling the output plot.
  par(mar=c(4,4.2,3,1.5))
  
  # 1 => plotting error(h) against the index of the training instance through the whole learning process.
  # 2 => plotting error(h) against the index of the training instances for the 1st 100 instances only.
  # 3 => plotting error(g) after applying threshold against the index of the training instance through the whole learning process.
  # 4 => plotting error(g) after applying threshold against the index of the training instance for the 1st 100 instances only.
  # 5 => plotting the class of the plant against its petal length for the provided 100 samples.
  # 6 => plotting the class of the plant against its sepal length for the provided 100 samples.
  
  if (selection == 1)
  {
    # all 6 iterations (1:600) (h)
    plot(c(1:(iterationsCounter*nrow(origData))), actualErrorValues, type="l", xlab="Index", ylab="Error",col="red",
         main = "Error value against Index over the whole learning process \n Using the output of the neural network directly without thresholding",
         cex.lab=1.2, cex.axis = 1.2, cex=1.2)
  } else if(selection == 2)
  {
    # first 100 instances (1:100) (h)
    plot(c(1:100), actualErrorValues[1:100], type="l", xlab="Index", ylab="Error", col="red",
         main = "Error value against the instance index ranging over 1:100 \n Using the output of the neural network directly without thresholding",
         cex.lab=1.2, cex.axis = 1.2, cex=1.2)
  } else if(selection == 3)
  {
    # all 6 iterations (1:600) (g)
    plot(c(1:(iterationsCounter*nrow(origData))), thresholdErrorValues, type="l", xlab="Index", ylab="Error", col="red",
         main = "Error value against Index over the whole learning process \n Using thresholding",
         cex.lab=1.2, cex.axis = 1.2, cex=1.2)
  } else if(selection == 4)
  {
    # first 100 instances (1:100) (g)
    plot(c(1:100), thresholdErrorValues[1:100], type="l", xlab="Index", ylab="Error", col="red",
         main = "Error value against the instance index ranging over 1:100 \n Using thresholding",
         cex.lab=1.2, cex.axis = 1.2, cex=1.2)
  }
  else if(selection == 5)
  {
    # first 100 instances (1:100) (g) (5.a) (Iris Class VS Petal Length)
    plot(as.matrix(origData[3]), speciesNumeric, xlab="Petal length", ylab="Iris Class (-1:Versicolor / 1:Setosa)",
         col='blue', main = "Iris class variable versus the petal length", cex.lab=1.2, cex.axis = 1.2, cex=1.2)
  } else if(selection == 6)
  {
    # first 100 instances (1:100) (g) (5.a) (Iris Class VS Sepal Length)
    plot(as.matrix(origData[2]), speciesNumeric, xlab="Sepal length", ylab="Iris Class (-1:Versicolor / 1:Setosa)",
         col='blue', main = "Iris class variable versus the sepal length", cex.lab=1.2, cex.axis = 1.2, cex=1.2)
  } else
  {
    print("Wrong input")
  }
}

# This function calculates the variance and correlation as described in the function. 
covarianceAndCorrelationFunction <- function()
{
  # Petal:
  # covariance between plant class and its petal length
  petalCov <- cov(speciesNumeric, origData$species)
  # correlation between plant class and its petal length
  petalCor <- cor(speciesNumeric, origData$species)
  print(paste("Petal covariance =", petalCov, " and Petal correlation =", petalCor))
  
  # Sepal:
  # covariance between plant class and its sepal length
  sepalCov <- cov(speciesNumeric, origData$petal)
  # correlation between plant class and its sepal length
  sepalCor <- cor(speciesNumeric, origData$petal)
  print(paste("Sepal covariance =", sepalCov, " and Sepal correlation =", sepalCor))
}
