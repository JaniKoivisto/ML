#run kmeans()

visualize_classification <- function(classification_data) {

  f1 <- as.numeric(levels(classification_data$population_density_people_per_square_km))[classification_data$population_density_people_per_square_km]
  f2 <- as.numeric(levels(classification_data$Internet_users_per_100_people))[classification_data$Internet_users_per_100_people]
  
  plot(f1, f2, col = "red", main = "kmeans results")
  
  #update classification_data
  classification_data[, 2] <- f1
  classification_data[, 3] <- f2
  
  #selects only data of class 1
  class1_f1 <- classification_data[classification_data[, 4] == "1", 2]
  class1_f2 <- classification_data[classification_data[, 4] == "1", 3]
  
  points(class1_f1, class1_f2, col = "violet", lwd = 3)
  
  #selects only data of class 2
  class2_f1 <- classification_data[classification_data[, 4] == "2", 2]
  class2_f2 <- classification_data[classification_data[, 4] == "2", 3]
  
  points(class2_f1, class2_f2, col = "green", lwd = 3)
  
  #selects only data of class 3
  class3_f1 <- classification_data[classification_data[, 4] == "3", 2]
  class3_f2 <- classification_data[classification_data[, 4] == "3", 3]
  
  points(class3_f1, class3_f2, col = "blue", lwd = 3)
  
  legend("bottomright", legend = c("class 1", "class 2", "class 3"), pch = c(1,1,1), # pch sets point symbol
         col = c("violet", "green", "blue"), pt.cex = c(2,2,2), pt.lwd = c(3,3,3))
  #print result
  print(classification_data)
  
  return(1)
}

readData <- function() {
  data <- read.csv2("test_data.csv")
  return(data)
}


kmeans <- function() {
  data <- readData()
  size <- dim(data)
  
  classification_column <- matrix(nrow = size[1], ncol = 1);
  classification_data <- cbind(data, classification_column)
  
  numeric_data <- matrix( nrow = size[1], ncol = 2)
  numeric_data <- data[1:size[1], 2:3]
  
  numeric_data <- apply(numeric_data, 2, as.numeric)
  
  normalize1 <- max(numeric_data[, 1])
  normalize2 <- max(numeric_data[, 2])
  
  denormalize<-c(normalize1, normalize2)
  
  numeric_data[, 1] <- numeric_data[, 1]/normalize1
  numeric_data[, 2] <- numeric_data[, 2]/normalize2
  
  numericDataSize <- dim(numeric_data)
  
  class1 <- apply(numeric_data, 2, min)
  class2 <- apply(numeric_data, 2, max)
  class3 <- apply(numeric_data, 2, mean)
  
  featureVectors <- list(class1, class2, class3)

  class1_prev <- c(-10, -10)
  class2_prev <- c(-10, -10)
  class3_prev <- c(-10, -10)
  iterations <- 0
  
  while (iterations < 20) {
    if (featureVectors[[1]] == class1_prev && featureVectors[[2]] == class2_prev && featureVectors[[3]] == class3_prev) {
       break
    } else

      for (i in 1:numericDataSize[1]) {
        distance_data <- matrix( nrow = 3, ncol = 2)
        distanceDataSize <- dim(distance_data)
        
        distance_data[1,2] <- 1
        distance_data[2,2] <- 2
        distance_data[3,2] <- 3
        
        dataVector <- c(numeric_data[i,1], numeric_data[i,2])
        
        for (j in 1:distanceDataSize[1]) {
          distance_data[j,1] <- euclidean_distance(dataVector, featureVectors[[j]])
        }
        
        closestClass <- head(distance_data[order(distance_data[,1], decreasing = FALSE),], 1)[1,2]
        classification_data[i,4] <- closestClass
        
      }
    
    #update data
    class1_prev <- class1
    class2_prev <- class2
    class3_prev <- class3
    
    class1_data <- classification_data[classification_data[, 4] == "1",]
    class1[1] <- mean(as.numeric(levels(class1_data[, 2])[class1_data[, 2]])/normalize1)
    class1[2] <- mean(as.numeric(levels(class1_data[, 3])[class1_data[, 3]])/normalize2)
    
    class2_data <- classification_data[classification_data[, 4] == "2",]
    class2[1] <- mean(as.numeric(levels(class2_data[, 2])[class2_data[, 2]])/normalize1)
    class2[2] <- mean(as.numeric(levels(class2_data[, 3])[class2_data[, 3]])/normalize2)
    
    class3_data <- classification_data[classification_data[, 4] == "3",]
    class3[1] <- mean(as.numeric(levels(class3_data[, 2])[class3_data[, 2]])/normalize1)
    class3[2] <- mean(as.numeric(levels(class3_data[, 3])[class3_data[, 3]])/normalize2)
    
    featureVectors <- list(class1, class2, class3)
 
    iterations <- iterations + 1
  }
  
  visualize_classification(classification_data)
  message("Iterations: ", iterations)
  return(1)
}

euclidean_distance <- function(dataVector, trainVector) {
  sqrt((dataVector[1] - trainVector[1])^2 + (dataVector[2] - trainVector[2])^2)
}






