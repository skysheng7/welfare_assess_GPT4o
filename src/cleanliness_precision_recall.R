#library(dplyr)
library(ggplot2)
library(viridis)

setwd("C:/Users/skysheng/Desktop/lameness_GPT4V/results_welfare_assess/cleanliness")
results <- read.csv("welfare_assess_cleanliness_gpt4o.csv", header = TRUE)

# Convert scores to binary format if they are not already (e.g., from character to numeric)
results$true_score <- as.numeric(results$true_score)
results$predict_score <- as.numeric(results$predict_score)

############################ percentage of accurate prediction ############################
# as large language models output are probability distribution, we randomly run the 
# same test image multiple times to get GPT-4o's assessment results. 
# we calculate the frequency of each image being predicted as score 0 and 2
# for each image, we get an accuracy measure, that reprecent percentage of times
# GPT-4o made the right prediction
hindleg <- results[which(results$assess_area == "hindleg cleanliness"),]
summary_df <- summarize_scores(results)
summary_df$inaccurate_pct <- 1-summary_df$accurate_predict_pct
overall_accuracy <- mean(summary_df$accurate_predict_pct)
metrics_list <- weighted_precision_recall(summary_df)
overall_precision <- metrics_list$precision
overall_recall <- metrics_list$recall


##################### accuracy, precision, recall for each group###########################
all_metrics_df <- calculate_metrics(results)
# Plot heatmaps for each metric
<<<<<<< HEAD
accuracy_heatmap <- plot_heatmap(all_metrics_df, "Accuracy")
precision_heatmap <- plot_heatmap(all_metrics_df, "Precision")
recall_heatmap <- plot_heatmap(all_metrics_df, "Recall")
=======
accuracy_heatmap <- plot_heatmap(all_metrics_df, "accuracy")
precision_heatmap <- plot_heatmap(all_metrics_df, "precision")
recall_heatmap <- plot_heatmap(all_metrics_df, "recall")
>>>>>>> 5d1fe4b29a735dc4996beee70911d3cb34178fda

# Print or save the plots
print(accuracy_heatmap)
print(precision_heatmap)
print(recall_heatmap)