#--------------------------------------#
# ACTIVATION DES LIRAIRIES NECESSAIRES #
#--------------------------------------#

#install.packages("C50")
library(C50)
#install.packages("randomForest")
library(randomForest)
#install.packages("e1071")
library(e1071)
#install.packages("naivebayes")
library(naivebayes)
#install.packages("nnet")
library(nnet)
#install.packages("kknn")
library(kknn)
#install.packages("ggplot2")
library(ggplot2)
#install.packages("ROCR")
library(ROCR)

#-------------------------#
# PREPARATION DES DONNEES #
#-------------------------#

# Chargement des donnees
produit <- read.csv("C:/Users/OneDrive/Bureau/ESTIA - 2020/TP5/M2_MIAGE_Data_Produit.csv", header = TRUE, sep = "\t", dec = ".")

# Nuage de points Revenus, Enfants, Produit
qplot(Revenus, Enfants, data=produit, color=Produit) + geom_jitter(height = 0.3)

# Création de la variable Quotient_Familial
produit$Quotient_Familial <- ifelse(produit$Enfants==0, produit$Revenus, produit$Revenus/produit$Enfants)

# Ecriture du fichier avec Quotient_Familial
write.table(produit, "Data_Produit_QF.csv", sep="\t", dec=".", row.names=FALSE)

# Creation des ensembles d'apprentissage et de test
produit_QF_EA <- produit[1:400,]
produit_QF_ET <- produit[401:600,]

# Suppression de la variable ID (identifiant)
produit_QF_EA <- subset(produit_QF_EA, select = -ID)
produit_QF_ET <- subset(produit_QF_ET, select = -ID) 

#--------------------#
# ARBRE DE DECISION  #
#--------------------#

# Apprentissage du classifeur de type arbre de décision
dt <- C5.0(Produit~., produit_QF_EA)
dt

# Test du classifieur : classe predite
dt_class <- predict(dt, produit_QF_ET, type="class")
dt_class
table(dt_class)

# Matrice de confusion
table(produit_QF_ET$Produit, dt_class)

# Test du classifieur : probabilites pour chaque prediction
dt_prob <- predict(dt, produit_QF_ET, type="prob")

# Courbe ROC
dt_pred <- prediction(dt_prob[,2], produit_QF_ET$Produit)
dt_perf <- performance(dt_pred,"tpr","fpr")
plot(dt_perf, col = "burlywood4")

# Calcul de l'AUC
dt_auc <- performance(dt_pred, "auc")
attr(dt_auc, "y.values")

#----------------#
# RANDOM FORESTS #
#----------------#

# Apprentissage du classifeur de type foret aleatoire
rf <- randomForest(Produit~., produit_QF_EA)
rf

# Test du classifieur : classe predite
rf_class <- predict(rf,produit_QF_ET, type="response")
rf_class
table(rf_class)

# Matrice de confusion
table(produit_QF_ET$Produit, rf_class)

# Test du classifieur : probabilites pour chaque prediction
rf_prob <- predict(rf, produit_QF_ET, type="prob")
# L'objet genere est une matrice 
rf_prob

# Courbe ROC
rf_pred <- prediction(rf_prob[,2], produit_QF_ET$Produit)
rf_perf <- performance(rf_pred,"tpr","fpr")
plot(rf_perf, add = TRUE, col = "red")

# Calcul de l'AUC
rf_auc <- performance(rf_pred, "auc")
attr(rf_auc, "y.values")

#-------------------------#
# SUPPORT VECTOR MACHINES #
#-------------------------#

# Apprentissage du classifeur de type svm
svm <- svm(Produit~., produit_QF_EA, probability=TRUE)
svm

# Test du classifieur : classe predite
svm_class <- predict(svm, produit_QF_ET, type="response")
svm_class
table(svm_class)

# Matrice de confusion
table(produit_QF_ET$Produit, svm_class)

# Test du classifieur : probabilites pour chaque prediction
svm_prob <- predict(svm, produit_QF_ET, probability=TRUE)

# L'objet genere est de type specifique aux svm
svm_prob

# Recuperation des probabilites associees aux predictions
svm_prob <- attr(svm_prob, "probabilities")

# Conversion en un data frame 
svm_prob <- as.data.frame(svm_prob)

# Courbe ROC sur le meme graphique
svm_pred <- prediction(svm_prob$Oui, produit_QF_ET$Produit)
svm_perf <- performance(svm_pred,"tpr","fpr")
plot(svm_perf, add = TRUE, col = "blue")

# Calcul de l'AUC
svm_auc <- performance(svm_pred, "auc")
attr(svm_auc, "y.values")

#-------------#
# NAIVE BAYES #
#-------------#

# Apprentissage du classifeur de type naive bayes
nb <- naive_bayes(Produit~., produit_QF_EA)
nb

# Test du classifieur : classe predite
nb_class <- predict(nb, produit_QF_ET, type="class")
nb_class
table(nb_class)

# Matrice de confusion
table( produit_QF_ET$Produit, nb_class)

# Test du classifieur : probabilites pour chaque prediction
nb_prob <- predict(nb, produit_QF_ET, type="prob")

# L'objet genere est une matrice
nb_prob

# Courbe ROC
nb_pred <- prediction(nb_prob[,2], produit_QF_ET$Produit)
nb_perf <- performance(nb_pred,"tpr","fpr")
plot(nb_perf, add = TRUE, col = "darkgreen")

# Calcul de l'AUC
nb_auc <- performance(nb_pred, "auc")
attr(nb_auc, "y.values")

#-----------------#
# NEURAL NETWORKS #
#-----------------#

# Apprentissage du classifeur de type perceptron monocouche
nn <- nnet(Produit~., produit_QF_EA, size=12)
nn

# Test du classifieur : classe predite
nn_class <- predict(nn, produit_QF_ET, type="class")
nn_class
table(nn_class)

# Matrice de confusion
table(produit_QF_ET$Produit, nn_class)

# Test du classifieur : probabilites pour chaque prediction
nn_prob <- predict(nn, produit_QF_ET, type="raw")

# L'objet genere est un vecteur des probabilites de prediction "Oui"
nn_prob

# Courbe ROC
nn_pred <- prediction(nn_prob[,1], produit_QF_ET$Produit)
nn_perf <- performance(nn_pred,"tpr","fpr")
plot(nn_perf, add = TRUE, col = "black")

# Calcul de l'AUC
nn_auc <- performance(nn_pred, "auc")
attr(nn_auc, "y.values")

#---------------------#
# K-NEAREST NEIGHBORS #
#---------------------#

# Apprentissage et test simultanes du classifeur de type k-nearest neighbors
knn <- kknn(Produit~., produit_QF_EA, produit_QF_ET)

# Resultat : classe predite et probabilites de chaque classe pour chaque instance de test
summary(knn)

# Matrice de confusion
table(produit_QF_ET$Produit, knn$fitted.values)

# Conversion des probabilites en data frame
knn_prob <- as.data.frame(knn$prob)

# Courbe ROC
knn_pred <- prediction(knn_prob$Oui, produit_QF_ET$Produit)
knn_perf <- performance(knn_pred,"tpr","fpr")
plot(knn_perf, add = TRUE, col = "darkmagenta")

# Calcul de l'AUC
knn_auc <- performance(knn_pred, "auc")
attr(knn_auc, "y.values")

