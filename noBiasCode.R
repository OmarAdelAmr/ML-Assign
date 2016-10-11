setwd('/home/omar/Desktop/ML_Assign')
origData <- read.csv2('/home/omar/Desktop/ML_Assign/iris_sub_21655.csv', sep = ',', header = TRUE, stringsAsFactors = FALSE)
origData$petal <- as.numeric(origData$petal)
origData$species <- as.numeric(origData$species)

speciesFactor <- origData$X
levels(speciesFactor) <- c(levels(speciesFactor), 1, -1)
speciesFactor[speciesFactor=='setosa'] <- 1
speciesFactor[speciesFactor=='versicolor'] <- -1
speciesFactor <- as.numeric(as.character(speciesFactor))