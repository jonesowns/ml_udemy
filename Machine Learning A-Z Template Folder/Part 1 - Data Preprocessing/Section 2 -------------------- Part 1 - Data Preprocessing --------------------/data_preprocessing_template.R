# Data Preprocessing

# Importing the libraries
# nothing to do


# Importing the dataset
dataset = read.csv('Data.csv')

# make the matrix of features
# nothing to do


# Taking care of Missing data

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)


dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)


# Encoding the Catogorical data
dataset$Country = factor(dataset$Country, 
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3)
                         )

dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes' ),
                           labels = c(0,1)
                           )

#Split into training and test sets
# install.packages('caTools') 
# above line is to import/install packages

set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)



#Feature Scaling
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])








