library(tidyverse)
library(ParamHelpers)
library(mlr)

train_data <- read_csv("../input/train.csv")
test_data <- read_csv("../input/test.csv")

train <- sample(train_data)
train$Survived <- as.factor(train$Survived)
test <- test_data %>% mutate(Survived = 1)
test[is.na(test)] <- 7
test$Survived[10] <- 0
test$Survived <- as.factor(test$Survived)


clean_data = function(raw_data){
    return (raw_data %>% 
        mutate( Upper = case_when(Pclass == 3 ~1, TRUE ~0),
                Middle = case_when(Pclass == 2 ~1, TRUE ~0),
                Lower = case_when(Pclass == 1 ~1, TRUE ~0),
                Male = case_when(Sex == "male" ~1, TRUE ~0),
                Female = case_when(Sex == "female" ~1, TRUE ~0),
                LocationQ = case_when(Embarked == "Q" ~1, TRUE ~0),
                LocationS = case_when(Embarked == "S" ~1, TRUE ~0),
                LocationC = case_when(Embarked == "C" ~1, TRUE ~0),
                Baby = case_when(Age < 0 ~1, TRUE ~0),
                Mr = case_when(grepl("Mr.", Name, fixed=TRUE) ~1, TRUE ~0),
                Mrs = case_when(grepl("Mrs.", Name, fixed=TRUE) ~1, TRUE ~0),
                Miss = case_when(grepl("Miss", Name, fixed=TRUE) ~1, TRUE ~0),
                Master = case_when(grepl("Master", Name, fixed=TRUE) ~1, TRUE ~0)) %>%
        mutate( Other = case_when(Mr != 1 & Mrs != 1 & Miss != 1 & Master != 1 ~1, TRUE ~0)) %>%
        mutate( Age = replace(Age, is.na(Age) & Mr == 1, 32.36),
                Age = replace(Age, is.na(Age) & Mrs == 1, 35.36),
                Age = replace(Age, is.na(Age) & Miss == 1, 21.77),
                Age = replace(Age, is.na(Age) & Master == 1, 4.57),
                Age = replace(Age, is.na(Age) & Other == 1, 29.70)) %>%
        select(-c(PassengerId, Pclass, Name, Sex, Ticket, Cabin, Embarked)))
}

#a <- clean_data(train_data) %>% 
#    mutate(a = case_when(Other = 1 & !is.na(Age) ~Age, TRUE ~NA_real_)) %>%
#    select(a) 
#mean(data.matrix(a), na.rm = TRUE)
##Average Age of the Titles
##Mr. 32.36
##Mrs. 35.36
##Miss 21.77
##Master 4.57

## coverts model to a csv submission file
to_csv <- function(submit, name){
    submit <- data.frame(submit) %>% 
    mutate(PassengerId = id + 891, Survived = response) %>%
    select(-c(id, truth, response))
    
    submit$Survived <- as.numeric(as.character(submit$Survived))
    write.csv(submit, paste(name, ".csv", sep = ""), row.names = F)
}

## Followed Guide from https://www.analyticsvidhya.com/blog/2016/08/practicing-machine-learning-techniques-in-r-with-mlr-package/
train_task <- makeClassifTask(data = clean_data(train), target = "Survived", positive = 1)
test_task <- makeClassifTask(data = clean_data(test), target = "Survived", positive = 1)

##logistic regression
logistic.learner <- makeLearner("classif.logreg",predict.type = "response")
cv.logistic <- crossval(learner = logistic.learner,task = train_task,iters = 3,stratify = TRUE,measures = acc,show.info = F)
fmodel <- train(logistic.learner,train_task)
getLearnerModel(fmodel)
fpmodel <- predict(fmodel, test_task)

to_csv(fpmodel, "logistic_regression")


##SVM
getParamSet("classif.ksvm")
ksvm <- makeLearner("classif.ksvm", predict.type = "response")
pssvm <- makeParamSet(
    makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), 
    makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)))
ctrl <- makeTuneControlGrid()
set_cv <- makeResampleDesc("CV",iters = 3L)
res <- tuneParams(ksvm, task = train_task, resampling = set_cv, par.set = pssvm, control = ctrl, measures = acc)
t.svm <- setHyperPars(ksvm, par.vals = res$x)
par.svm <- train(ksvm, train_task)
predict.svm <- predict(par.svm, test_task)

to_csv(predict.svm, "svm")

##random forestn
getParamSet("classif.randomForest")
rf <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf$par.vals <- list(
    importance = TRUE
)
rf_param <- makeParamSet(
    makeIntegerParam("ntree",lower = 50, upper = 500),
    makeIntegerParam("mtry", lower = 3, upper = 10),
    makeIntegerParam("nodesize", lower = 10, upper = 50)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
set_cv <- makeResampleDesc("CV",iters = 3L)
rf_tune <- tuneParams(learner = rf, resampling = set_cv, task = train_task, par.set = rf_param, control = rancontrol, measures = acc)
rf.tree <- setHyperPars(rf, par.vals = rf_tune$x)
rforest <- train(rf.tree, train_task)
rfmodel <- predict(rforest, test_task)

to_csv(rfmodel, "rfmodel")



