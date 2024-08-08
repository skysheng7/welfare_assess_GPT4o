library(dplyr)

setwd("C:/Users/skysheng/Desktop/lameness_GPT4V/results_welfare_assess/cleanliness")
results <- read.csv("welfare_assess_cleanliness_gpt4o.csv", header = TRUE)

# Convert scores to binary format if they are not already (e.g., from character to numeric)
results$true_score <- as.numeric(results$true_score)
results$predict_score <- as.numeric(results$predict_score)

############################ percentage of predicted score ############################
# as large language models output are probability distribution, we randomly run the 
# same test image multiple times to get GPT-4o's assessment results. 
# we calculate the frequency of each image being predicted as score 0 and 2
hindleg <- results[which(results$assess_area == "hindleg cleanliness"),]
summary_df <- summarize_scores(hindleg)



############################ overall precision & recall ############################
original <- results[which(results$treatment == "original"),]
# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
TP <- sum(original$predict_score == 2 & original$true_score == 2)
FP <- sum(original$predict_score == 2 & original$true_score == 0)
FN <- sum(original$predict_score == 0 & original$true_score == 2)

# Calculate Precision and Recall
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

# Print Precision and Recall
cat("Overall Precision:", precision, "\n")
cat("Overall Recall:", recall, "\n")


############################ hindleg precision & recall ############################
hindleg <- original[which(original$assess_area == "hindleg cleanliness"),]
# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
TP <- sum(hindleg$predict_score == 2 & hindleg$true_score == 2)
FP <- sum(hindleg$predict_score == 2 & hindleg$true_score == 0)
FN <- sum(hindleg$predict_score == 0 & hindleg$true_score == 2)

# Calculate Precision and Recall
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

# Print Precision and Recall
cat("Hindleg Precision:", precision, "\n")
cat("Hindleg Recall:", recall, "\n")


############################ hindleg & hindquarter precision & recall ############################
hindleg_hindquarter <- original[which((original$assess_area == "hindleg cleanliness") | (original$assess_area == "hindquarter cleanliness") ),]
# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
TP <- sum(hindleg_hindquarter$predict_score == 2 & hindleg_hindquarter$true_score == 2)
FP <- sum(hindleg_hindquarter$predict_score == 2 & hindleg_hindquarter$true_score == 0)
FN <- sum(hindleg_hindquarter$predict_score == 0 & hindleg_hindquarter$true_score == 2)

# Calculate Precision and Recall
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

# Print Precision and Recall
cat("Hindleg & hindquarter Precision:", precision, "\n")
cat("Hindleg & hindquarter Recall:", recall, "\n")