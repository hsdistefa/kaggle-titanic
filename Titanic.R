library('ggplot2')
library('ggthemes')
library('scales')
library('dplyr')
library('mice')
library('randomForest')

train <- read.csv('input/train.csv', stringsAsFactors = F)
test  <- read.csv('input/test.csv', stringsAsFactors = F)

full  <- bind_rows(train, test)

# Grab title from passenger names
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# Group similar titles
rare_title <- c('Dona', 'Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
full$Title[full$Title == 'Mlle'] <- 'Miss'
full$Title[full$Title == 'Ms'] <- 'Miss'
full$Title[full$Title == 'Mme'] <- 'Mrs'
full$Title[full$Title %in% rare_title] <- 'Rare Title'

# Grab last names
full$Surname <- sapply(full$Name,
                       function(x) strsplit(x, split='[,.]')[[1]][1])

# Create family size variable
full$Fsize <- full$SibSp + full$Parch + 1

# Create family variable
full$Family <- paste(full$Surname, full$Fsize, sep='_')

# Discretize family size
full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

# Create a deck variable
full$Deck <- factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

# Get rid of missing embarkment passengerids to look at the others
embark_fare <- subset(full, PassengerId != 62 & PassengerId != 830)

# Most likely embarked from 'C' looking at fare and pclass
full$Embarked[c(62,830)] <- 'C'

# Most likely fare around $8.05 looking at embarked and pclass
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)

# Make variables factors into factors
factor_vars <- c('PassengerId', 'Pclass', 'Sex', 'Embarked', 'Title', 'Surname', 'Family', 'FsizeD')
full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# Set a random seed
set.seed(129)

# Perform mice imputation on age, excluding less-useful variables
mice_mod <- mice(full[, !names(full) %in% c('PassengerId', 'Name', 'Ticket', 'Cabin', 'Family', 
'Surname', 'Survived')], method='rf')

mice_output <- complete(mice_mod)
full$Age <- mice_output$Age

# Create the column child
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

# Add mother variable
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

# Factorize added variables
full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)

# Split back into train and test set
train <- full[1:891,]
test <- full[892:1309,]

# Build ranfom forest model
set.seed(754)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                                            Fare + Embarked + Title + FsizeD +
                                            Child + Mother, data = train)

# Variable importance
importance <- importance(rf_model)
var_importance <- data.frame(Variables = row.names(importance),
                             Importance = round(importance[,'MeanDecreaseGini'],2))
rank_importance <- var_importance %>% mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Prediction
prediction <- predict(rf_model, test)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)