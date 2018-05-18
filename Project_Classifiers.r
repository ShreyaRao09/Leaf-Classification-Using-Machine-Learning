#Leaf Classification using Bagging, Random Forest, K-Nearest Neighbor, Naive Bayes and Support Vector Machine classifiers
# Version 1.0: 11/27/2016
require(e1071)
require(caret)
require(kernlab)
require(mlbench)
require(adabag)


myData <- read.csv("train.csv")
test_final<-read.csv("test.csv")

#shifting "species" to the last column
species<-myData[,2]
myData<-myData[,-2]
myData = cbind(myData,species)

#removing columns manually
myData<-myData[,-193]
myData<-myData[,-192]
myData<-myData[,-190]
myData<-myData[,-189]
myData<-myData[,-185]
myData<-myData[,-184]
myData<-myData[,-181]
myData<-myData[,-180]
myData<-myData[,-171]
myData<-myData[,-170]
myData<-myData[,-165]
myData<-myData[,-162]
myData<-myData[,-161]
myData<-myData[,-150]
myData<-myData[,-147]
myData<-myData[,-145]
myData<-myData[,-144]
myData<-myData[,-143]
myData<-myData[,-139]
myData<-myData[,-63]
myData<-myData[,-62]
myData<-myData[,-61]
myData<-myData[,-53]
myData<-myData[,-44]
myData<-myData[,-38]
myData<-myData[,-35]
myData<-myData[,-31]
myData<-myData[,-24]
myData<-myData[,-17]
myData<-myData[,-10]
myData<-myData[,-9]

#scalling the data
species<-myData[,163]
myData<-myData[,-163]
maxs<-apply(myData,2,max)
mins<-apply(myData,2,min)
myData<-as.data.frame(scale(myData,center = mins,scale = maxs-mins))
myData = cbind(myData,species)

#splitting data into training and testing sets to determine accuracy
Index <- sample(1:nrow(myData), 0.8 * nrow(myData))
train <- myData[Index, ]
test <- myData[-Index, ]
id_train<-train[,163]
id_test<-test[,163]
train<-train[,-1]
test<-test[,-1]

column_name<-names(train[,-162])
formula_leafdata<-as.formula(paste("species~",paste(column_name,collapse = "+"),collapse = NULL))
fitControl <- trainControl(method = "repeatedcv",number = 10,repeats = 5)

#Bagging
method<-"Bagging"
leaf_bagging_model <- train(formula_leafdata,data = train , method = "treebag", trControl = fitControl,verbose = FALSE)
leaf_bagging_output <- predict(leaf_bagging_model,test)

Bagging_accuracy<-mean(leaf_bagging_output==test[,162])
cat("Using Classifier ", method,": Accuracy = ", Bagging_accuracy*100,"% \n")

#Random Forest
method<-"Random Forest"
leaf_rf_model <- train(formula_leafdata,data = train , method = "rf", trControl = fitControl,verbose = FALSE)
leaf_rf_output <- predict(leaf_rf_model,test)

RF_accuracy<-mean(leaf_rf_output==test[,162])
cat("Using Classifier ", method,": Accuracy = ", RF_accuracy*100,"% \n")

#K-Nearest Neighbour
method<-"KNN"
leaf_knn_Model <- knn3(formula_leafdata,train,k=3)

predictions_knn <- predict(leaf_knn_Model, test[,-162], type="class")

KNN_accuracy<-mean(predictions_knn==test[,162])
cat("Using Classifier ", method,": Accuracy = ", KNN_accuracy*100,"% \n")

#Naive Bayes
method<-"Naive Bayes"
leaf_nb_Model <- naiveBayes(formula_leafdata,train)

predictions_nb <- predict(leaf_nb_Model, test[,-162], type="class")

nb_accuracy<-mean(predictions_nb==test[,162])
cat("Using Classifier ", method,": Accuracy = ", nb_accuracy*100,"% \n")

#Support Vector Machine
method<-"SVM"
leaf_svm_Model <- ksvm(formula_leafdata,data=train,kernel="rbfdot")

predictions_svm <- predict(leaf_svm_Model, test[,-162])

svm_accuracy<-mean(predictions_svm==test[,162])
cat("Using Classifier ", method,": Accuracy = ", svm_accuracy*100,"% \n")


#predicting values of the unlabled data as given in the competition
test_final<-test_final[,-193]
test_final<-test_final[,-192]
test_final<-test_final[,-190]
test_final<-test_final[,-189]
test_final<-test_final[,-185]
test_final<-test_final[,-184]
test_final<-test_final[,-181]
test_final<-test_final[,-180]
test_final<-test_final[,-171]
test_final<-test_final[,-170]
test_final<-test_final[,-165]
test_final<-test_final[,-162]
test_final<-test_final[,-161]
test_final<-test_final[,-150]
test_final<-test_final[,-147]
test_final<-test_final[,-145]
test_final<-test_final[,-144]
test_final<-test_final[,-143]
test_final<-test_final[,-139]
test_final<-test_final[,-63]
test_final<-test_final[,-62]
test_final<-test_final[,-61]
test_final<-test_final[,-53]
test_final<-test_final[,-44]
test_final<-test_final[,-38]
test_final<-test_final[,-35]
test_final<-test_final[,-31]
test_final<-test_final[,-24]
test_final<-test_final[,-17]
test_final<-test_final[,-10]
test_final<-test_final[,-9]
id_test_num<-test_final[,1]
test_final<-test_final[,-1]

#scale
maxs<-apply(test_final,2,max)
mins<-apply(test_final,2,min)
test_final<-as.data.frame(scale(test_final,center = mins,scale = maxs-mins))

#predicting the final output using KNN Classifier
final_knn_output <- predict(leaf_knn_Model,test_final)

result<-cbind.data.frame(id,final_knn_output)
write.csv(result,file="sample_submission.csv")
