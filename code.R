setwd('/home/omar/Desktop/ML_Assign')
origData <- read.csv2('/home/omar/Desktop/ML_Assign/iris_sub_21655.csv', sep = ',',
                      header = TRUE, stringsAsFactors = FALSE)
origData$petal <- as.numeric(origData$petal)
origData$species <- as.numeric(origData$species)
speciesNumeric <- origData$X
levels(speciesNumeric) <- c(levels(speciesNumeric), 1, -1)
speciesNumeric[speciesNumeric=='setosa'] <- 1
speciesNumeric[speciesNumeric=='versicolor'] <- -1
speciesNumeric <- as.numeric(as.character(speciesNumeric))

bias <- (-1); biasWeight <- 0.05
sepalWeight <- -0.02; petalWeight <- 0.02
flag <- TRUE; iterationsCounter <- 0
learningRate <- 0.2

actualErrorValues <- vector()
thresholdErrorValues <- vector()

training <- function()
{
  while(flag){
    iterationsCounter <<- iterationsCounter + 1; flag <<- FALSE
    for (count in 1:nrow(origData)) {
      sepal <- origData[count, 2]; petal <- origData[count, 3]
      h <- (bias * biasWeight) + (sepal * sepalWeight) + (petal * petalWeight)
      g <- thresholdFunction(h)
      correctValue <- 0
      
      if (origData[count, 4] == 'setosa'){
        correctValue <- 1
        if (g <= 0){ 
          flag <<- TRUE
          updateWeights(g, 1, petal, sepal)
        }
      } else{
        correctValue <- -1
        if (g > 0){ 
          flag <<- TRUE
          updateWeights(g, -1, petal, sepal)
        }
      }
      
      actualErrorValues[(iterationsCounter - 1) * 100 + count] <<- h - correctValue
      thresholdErrorValues[(iterationsCounter - 1) * 100 + count] <<- g - correctValue
    }
  }
  print(length(actualErrorValues))
  print(paste("Training is finished successfully with learning rate = (", 
              learningRate, "). Number of iterations = ", iterationsCounter))
}

thresholdFunction <- function(hValue)
{
  if (hValue > 0){
    return (1)
  }
  return (-1)
}

updateWeights <- function(gValue, tValue, petal, sepal)
{
  biasWeight <<- (biasWeight - (learningRate * (gValue - tValue) * bias))
  petalWeight <<- (petalWeight - (learningRate * (gValue - tValue) * petal))
  sepalWeight <<- (sepalWeight - (learningRate * (gValue - tValue) * sepal))
}

testing <- function(inSepal, inPetal)
{
  y = (bias * biasWeight) + (inSepal * sepalWeight) + (inPetal * petalWeight)
  if (y > 0){
    print ("Setosa")
  }else{
    print("Versicolor")
  }
}

plottingFunction <- function(selection)
{
  
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

covarianceAndCorrelationFunction <- function()
{
  # Petal
  petalCov <- cov(speciesNumeric, origData$species)
  petalCor <- cor(speciesNumeric, origData$species)
  print(paste("Petal covariance =", petalCov, " and Petal correlation =", petalCor))
  # Sepal
  sepalCov <- cov(speciesNumeric, origData$petal)
  sepalCor <- cor(speciesNumeric, origData$petal)
  print(paste("Sepal covariance =", sepalCov, " and Sepal correlation =", sepalCor))
}
