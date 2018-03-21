#run knn()

read_data <- function() {
  data <- read.csv2("test_data.csv")
  
  f1 <- as.numeric(levels(data$population_density_people_per_square_km))[data$population_density_people_per_square_km]
  f2 <- as.numeric(levels(data$Internet_users_per_100_people))[data$Internet_users_per_100_people]
  
  training_data <- matrix(nrow = 6, ncol = 3)
  
  # initialize training data of classes
  training_data[1, 1:3] <- c(10, 90, 1)
  training_data[2, 1:3] <- c(20, 100, 1)
  
  #class2
  training_data[3, 1:3] <- c(250, 90, 2)
  training_data[4, 1:3] <- c(330, 90, 2)
  
  #class3
  training_data[5,1:3] <- c(100, 50, 3)
  training_data[6,1:3] <- c(150, 20, 3)
  
  data_list<-list(data=data, training_data=training_data)
  
  return(data_list)
}

visualize_classification<-function(classification_data){
  f1 <- as.numeric(levels(classification_data$population_density_people_per_square_km))[classification_data$population_density_people_per_square_km]
  f2 <- as.numeric(levels(classification_data$Internet_users_per_100_people))[classification_data$Internet_users_per_100_people]
  
  #unclassified points will be red
  plot(f1, f2, col="red", main="k-nn results")
  
  #update
  classification_data[, 2] <- f1
  classification_data[, 3] <- f2
  
  #selects only data of class 1
  class1_f1 <- classification_data[classification_data[, 4] == "1", 2]
  class1_f2 <- classification_data[classification_data[, 4] == "1", 3]
  
  #visualizalises class 1
  points(class1_f1, class1_f2, col = "violet", lwd = 3)
  
  #selects only data of class 2
  class2_f1 <- classification_data[classification_data[, 4] == "2", 2]
  class2_f2 <- classification_data[classification_data[, 4] == "2", 3]
  
  #visualizalises class 2
  points(class2_f1, class2_f2, col="green", lwd = 3)
  
  #selects only data of class 3
  class3_f1 <- classification_data[classification_data[, 4] == "3", 2]
  class3_f2 <- classification_data[classification_data[, 4] == "3", 3]
  
  #visualizalises class 2
  points(class3_f1, class3_f2, col = "blue", lwd = 3)

  legend("bottomright", legend = c("class 1", "class 2", "class 3", "unclassified"), pch = c(1,1,1), # pch sets point symbol
         col = c("violet", "green", "blue", "red"), pt.cex = c(2,2,2), pt.lwd = c(3,3,3))
  
  #print result
  print(classification_data)
}

knn <- function() {
  data_list <- read_data()

  data <- data_list$data
  training_data <- data_list$training_data
  
  #pick feature vectors 
  f1 <- as.numeric(levels(data$population_density_people_per_square_km))[data$population_density_people_per_square_km]
  f2 <- as.numeric(levels(data$Internet_users_per_100_people))[data$Internet_users_per_100_people]
  
  #normalization factors (when two variables have different scale, normalization is usually needed)
  normalize1 <- max(f1)
  normalize2 <- max(f2)

  size <- dim(data)
  
  #creates an empty column
  classification <- matrix(nrow = size[1], ncol = 1)

  #final result matrix
  classification_data <- cbind(data, classification)
  numeric_data <- matrix(nrow = size[1], ncol = 2)
  
  #selects only columns 2 and 3) from data matrix
  numeric_data <- data[1:size[1], 2:3]
  numeric_data <- apply(numeric_data, 2, as.numeric)
  
  #normalize
  training_data[, 1] <- training_data[, 1] / normalize1
  training_data[, 2] <- training_data[, 2] / normalize2
  numeric_data[, 1] <- numeric_data[, 1] / normalize1
  numeric_data[, 2] <- numeric_data[, 2] / normalize2

  for (i in 1:size[1]) {
    trainingDataSize = dim(training_data)
    distances <- matrix(nrow = trainingDataSize[1], ncol = 1)
    
    copyOfTrainingMatrix <- training_data[1:trainingDataSize[1], 1:3]
    copyOfTrainingMatrixWithDistances <- cbind(copyOfTrainingMatrix, distances)
    dataVector <- c(numeric_data[i,1], numeric_data[i,2])
    
    for (j in 1:trainingDataSize[1]) {
      trainVector <- c(copyOfTrainingMatrix[j,1], copyOfTrainingMatrix[j,2])
      
      #calculate euclidean distance between vectors
      copyOfTrainingMatrixWithDistances[j,4] <- euclidean_distance(dataVector, trainVector)
      
      #sort data in ascending order according to distances
      sortedDistances <- copyOfTrainingMatrixWithDistances[order(copyOfTrainingMatrixWithDistances[,4], decreasing = FALSE),]
      firstThreeDistances <- head(sortedDistances, 3)
      
      #which class occurs the most within three shortest distances  
      mostFrequentClass <- head(names(sort(table(firstThreeDistances[,3]), decreasing = TRUE)), 1)
      countOfWinners <- length(which(firstThreeDistances[,3] == mostFrequentClass))

      if(countOfWinners > 1) {
        classification_data[i,4] <- mostFrequentClass
      }
      #unclassification rule
      if(length(unique(firstThreeDistances[,3])) == 3) {
        classification_data[i,4] <- 0
      }
      
    }
    
  }
  
  visualize_classification(classification_data)
  
}

euclidean_distance <- function(dataVector, trainVector) {
  sqrt((dataVector[1] - trainVector[1])^2 + (dataVector[2] - trainVector[2])^2)
}


