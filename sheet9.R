#Import data
library(tm)
library(SnowballC)
require(tm.corpus.Reuters21578)
data(Reuters21578)
rtCorpus <- Reuters21578

#1. Exploring Dataset
class(rtCorpus)                       
class(rtCorpus[[1]])                  
ID(rtCorpus[[1]])                     
meta(rtCorpus[[1]])                   
show(rtCorpus)                        
summary(rtCorpus)                     
inspect(rtCorpus)                     

#2. Split Corpus into train set and test set
#Train Set
query <- "LEWISSPLIT == 'TRAIN'"
rtTrain <- tm_filter(rtCorpus, FUN = sFilter, query)
#Test set
query2 <- "LEWISSPLIT == 'TEST'"
rtTest <- tm_filter(rtCorpus, FUN = sFilter, query2)

#3. Preprocessing
cleanCorpus <- function(corpus) {
  corpus <- tm_map(corpus, tolower)               
  corpus <- tm_map(corpus, removeNumbers)         
  corpus <- tm_map(corpus, removePunctuation)     
  corpus <- tm_map(corpus, stripWhitespace)       
  corpus <- tm_map(corpus, stemDocument)         
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  return(corpus)
}
rtTrain <- cleanCorpus(rtTrain)
rtTest <- cleanCorpus(rtTest)

#4.1 Create Trem-document matrix
rtdmTrain <- DocumentTermMatrix(rtTrain)
dim(rtdmTrain)
#4.2 TF-IDF
library(slam)
#Train Set
term_tfidf_Train <- tapply(rtdmTrain$v/row_sums(rtdmTrain)[rtdmTrain$i], rtdmTrain$j, mean) * log2(nDocs(rtdmTrain)/col_sums(rtdmTrain > 0))
summary(term_tfidf_Train)
#Remove documents with value = 0.09 which is atleast equal to the median 
rtdmTrain <- rtdmTrain[,term_tfidf_Train >= 0.09]
rtdmTrain <- rtdmTrain[row_sums(rtdmTrain) > 0,]
dim(rtdmTrain)
#4.3 Remove Sparse Terms
rtdmTrain <- removeSparseTerms(rtdmTrain, 0.996)  #Explain why 0.996?
dim(rtdmTrain)
#4.4Create dictionary from train set terms after preprocessing and apply to test set to preprocess. This is so that terms in test set match train set
TermDic <- colnames(rtdmTrain)        
rtdmTest <- DocumentTermMatrix(rtTest, list(dictionary=TermDic))
dim(rtdmTest)

#5. Topic Model
#Assume 30 topics
k <- 30
rtm30 <- LDA(rtdm, k=k, method = "VEM", control = list(seed = as.integer(Sys.time())))
#20 most common terms in each topic to infere theme/topic
rtTerms <- terms(rtm30, 10)
rtTerms <- as.data.frame(rtTerms)
rtTopics <- posterior(rtm30, rtdm)$topics
df.rtTopics <- as.data.frame(rtTopics)

#Apply dictionary to both train set and test set to create termed document matrix
rtdm2Train <- DocumentTermMatrix(rtTrain, list(dictionary=TermDic))

#7.1 Classification
#Create label dataframe from corpus
labelFUN <- function(topic, corpus) {
  label <- list()
  #Create vector for each topic
  for (i in 1:length(topic)){
    label[[i]] <- rep(0, length(corpus))
    #Check if topic exists in each document of corpus
    for (j in 1:length(corpus)) {
      if (topic[i] %in% meta(corpus[[j]], tag = 'Topics')) {
        label[[i]][j] <- 1
      }
    }
  }
  #Convert list to df
  labelDF <- do.call(cbind.data.frame, label)
  names(labelDF) <- topic
  return(labelDF)
}

#Vector to hold 10 important topics in corpus as class label
Topic <- c("earn", "acquisition", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn")

#Dataframe for class labels
labelDF <- labelFUN(Topic, rtCorpus)

library(RTextTools)
rtdmNew <- rbind(rtdmTrain, rtdmTest)

#Create container for each topic
container <- list()
analytics <- list()
for(i in 1:length(Topic)){
  container[[i]] <- create_container(rtdmNew, labelDF[rownames(rtdmNew), i], trainSize = 1:12861, testSize= 12862:19049, virgin=FALSE)
  
  #Train
  SVM <- train_model(container[[i]], "SVM")
  RF <- train_model(container[[i]], "RF")
  TREE <- train_model(container[[i]], "TREE")
  
  #Classify
  SVMCL <- classify_model(container[[i]], SVM)
  RFCL <- classify_model(container[[i]], RF)
  TREECL <- classify_model(container[[i]], TREE)
  
  #Analytics
  analytics[[i]] <- create_analytics(container[[i]], cbind(SVMCL, RFCL, TREECL))
}
container <- list()
analytics <- list()

#Topic 1
container[[1]] <- create_container(rtdmNew, labelDF[rownames(rtdmNew), 1], trainSize = 1:12861, testSize= 12862:19049, virgin=FALSE)
#Train
SVM <- train_model(container[[1]], "SVM")
RF <- train_model(container[[1]], "RF")
TREE <- train_model(container[[1]], "TREE")
#Classify
SVMCL <- classify_model(container[[1]], SVM)
RFCL <- classify_model(container[[1]], RF)
TREECL <- classify_model(container[[1]], TREE)
#Analytics
analytics[[1]] <- create_analytics(container[[1]], cbind(SVMCL, RFCL, TREECL))

#Topic 2
container[[2]] <- create_container(rtdmNew, labelDF[rownames(rtdmNew), 2], trainSize = 1:12861, testSize= 12862:19049, virgin=FALSE)
#Train
SVM <- train_model(container[[2]], "SVM")
RF <- train_model(container[[2]], "RF")
TREE <- train_model(container[[2]], "TREE")
#Classify
SVMCL <- classify_model(container[[2]], SVM)
RFCL <- classify_model(container[[2]], RF)
TREECL <- classify_model(container[[2]], TREE)
#Analytics
analytics[[2]] <- create_analytics(container[[2]], cbind(SVMCL, RFCL, TREECL))

#Topic 3
container[[3]] <- create_container(rtdmNew, labelDF[rownames(rtdmNew), 3], trainSize = 1:12861, testSize= 12862:19049, virgin=FALSE)
#Train
SVM <- train_model(container[[3]], "SVM")
RF <- train_model(container[[3]], "RF")
TREE <- train_model(container[[3]], "TREE")
#Classify
SVMCL <- classify_model(container[[3]], SVM)
RFCL <- classify_model(container[[3]], RF)
TREECL <- classify_model(container[[3]], TREE)
#Analytics
analytics[[3]] <- create_analytics(container[[3]], cbind(SVMCL, RFCL, TREECL))

#Topic 4
container[[4]] <- create_container(rtdmNew, labelDF[rownames(rtdmNew), 4], trainSize = 1:12861, testSize= 12862:19049, virgin=FALSE)
#Train
SVM <- train_model(container[[4]], "SVM")
RF <- train_model(container[[4]], "RF")
TREE <- train_model(container[[4]], "TREE")
#Classify
SVMCL <- classify_model(container[[4]], SVM)
RFCL <- classify_model(container[[4]], RF)
TREECL <- classify_model(container[[4]], TREE)
#Analytics
analytics[[4]] <- create_analytics(container[[4]], cbind(SVMCL, RFCL, TREECL))

#Evaluation criteia 
Precision <-c()
Recall <- c()
F1Score <- c()

#SVM Accuracy dataframe
for(i in 1:length(Topics) {
  Precision[i] <- summary(analytics[[i]])[1]
  Recall[i] <- summary(analytics[[i]])[2]
  F1Score[i] <- summary(analytics[[i]])[3]
}
scoreSVM <- data.frame(Topics, Precision, Recall, F1Score)

#RF Accuracy dataframe
for(i in 1:length(Topics) {
  Precision[i] <- summary(analytics[[i]])[4]
  Recall[i] <- summary(analytics[[i]])[5]
  F1Score[i] <- summary(analytics[[i]])[6]
}
scoreRF <- data.frame(Topics, Precision, Recall, F1Score)

#TREE Accuracy dataframe
for(i in 1:length(Topics) {
  Precision[i] <- summary(analytics[[i]])[7]
  Recall[i] <- summary(analytics[[i]])[8]
  F1Score[i] <- summary(analytics[[i]])[9]
}
scoreTREE <- data.frame(Topics, Precision, Recall, F1Score)

#Visualise
library(ggplot2)
#Plot SVM Measures
plotSVM <- data.frame(scoreSVM[1], stack(scoreSVM[2:4]))
colnames(plotSVM) <- c("Topics", "Measures", "Type")
ggplot(data=scoreSVM, aes(x=Topics, y=Measures, group=Type, color=Type)) + geom_line() + geom_point()

#Plot RF Measures
plotRF <- data.frame(scoreRF[1], stack(scoreRF[2:4]))
colnames(plotRF) <- c("Topics", "Measures", "Type")
ggplot(data=scoreRF, aes(x=Topics, y=Measures, group=Type, color=Type)) + geom_line() + geom_point()

#Plot RF Measures
plotTREE <- data.frame(scoreTREE[1], stack(scoreTREE[2:4]))
colnames(plotTREE) <- c("Topics", "Measures", "Type")
ggplot(data=scoreTREE, aes(x=Topics, y=Measures, group=Type, color=Type)) + geom_line() + geom_point()

#8. Clustering
rtdmNew

## Create matrix
rtdmClust <- as.matrix(rtdmNew)

### k-means (this uses euclidean distance)
m <- as.matrix(dtm_tfxidf)
rownames(m) <- 1:nrow(m)

### don't forget to normalize the vectors so Euclidean makes sense
norm_eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)
m_norm <- norm_eucl(m)


### cluster into 10 clusters
cl <- kmeans(m_norm, 10)
cl

table(cl$cluster)

### show clusters using the first 2 principal components
plot(prcomp(m_norm)$x, col=cl$cl)

findFreqTerms(dtm[cl$cluster==1], 50)
inspect(reuters[which(cl$cluster==1)])
library(proxy)
d <- dist(m, method="cosine")
hc <- hclust(d, method="average")
plot(hc)

cl <- cutree(hc, 50)
table(cl)
findFreqTerms(dtm[cl==1], 50)