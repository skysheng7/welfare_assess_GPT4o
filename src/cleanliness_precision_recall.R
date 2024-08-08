setwd("C:/Users/skysheng/Desktop/lameness_GPT4V/results_welfare_assess/cleanliness")
results <- read.csv("welfare_assess_cleanliness_gpt4o.csv", header = TRUE)

original <- results[which(results$treatment == "original"),]

# Convert scores to binary format if they are not already (e.g., from character to numeric)
original$true_score <- as.numeric(original$true_score)
original$predict_score <- as.numeric(original$predict_score)

############################ overall precision & recall ############################
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