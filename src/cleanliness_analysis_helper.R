# Define the summarization function
summarize_scores <- function(data) {
  # Initialize an empty list to store data frames for each unique test image
  results_list <- list()
  
  # Get unique identifiers for grouping
  unique_images <- unique(data[c("test_image", "assess_area", "treatment", "true_score", "true_note")])
  
  # Iterate over each unique image group
  for (i in seq_along(unique_images$test_image)) {
    # Filter data for the current test image
    subset_data <- data[data$test_image == unique_images$test_image[i] & data$treatment == unique_images$treatment[i], ]
    
    # Calculate frequencies of predict_score 0 and 2
    score_0_freq <- sum(subset_data$predict_score == 0, na.rm = TRUE)
    score_2_freq <- sum(subset_data$predict_score == 2, na.rm = TRUE)
    
    # Calculate the accuracy percentage
    accurate_predictions <- sum(subset_data$predict_score == subset_data$true_score, na.rm = TRUE)
    total_predictions <- nrow(subset_data)
    accurate_predict_pct <- (accurate_predictions / total_predictions) 
    
    # Combine the current row with frequencies and accuracy percentage into a new data frame
    current_result <- cbind(unique_images[i, ], predict_score_0_freq = score_0_freq, predict_score_2_freq = score_2_freq, accurate_predict_pct = accurate_predict_pct)
    
    # Append the result to the list
    results_list[[i]] <- current_result
  }
  
  # Combine all individual data frames into one data frame
  final_results <- do.call(rbind, results_list)
  return(final_results)
}

weighted_precision_recall <- function(data) {
  TP <- sum(data[which(data$true_score == 2), "accurate_predict_pct"])
  FP <- sum(data[which(data$true_score == 0), "inaccurate_pct"])
  FN <- sum(data[which(data$true_score == 2), "inaccurate_pct"])
  
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  
  return(list(precision = precision, recall = recall))
}

calculate_metrics <- function(data) {
  unique_combinations <- unique(data[c("assess_area", "treatment")])
  metrics_df <- data.frame(assess_area=character(), treatment=character(), accuracy=numeric(), precision=numeric(), recall=numeric(), stringsAsFactors = FALSE)
  
  for (i in seq(nrow(unique_combinations))) {
    subset_data <- data[data$assess_area == unique_combinations$assess_area[i] & data$treatment == unique_combinations$treatment[i],]
    summary_subset <- summarize_scores(subset_data)
    summary_subset$inaccurate_pct <- 1 - summary_subset$accurate_predict_pct
    
    # Calculate accuracy, precision, and recall
    accuracy <- mean(summary_subset$accurate_predict_pct)
    prec_rec_metrics <- weighted_precision_recall(summary_subset)
    
    # Append to the dataframe and ensure types are numeric
    metrics_df[nrow(metrics_df) + 1, ] <- c(unique_combinations$assess_area[i], unique_combinations$treatment[i], as.numeric(accuracy), as.numeric(prec_rec_metrics$precision), as.numeric(prec_rec_metrics$recall))
  }
  
  return(metrics_df)
}

plot_heatmap <- function(metrics_df, metric) {
  # Ensure the metric column is treated as numeric
  metrics_df[[metric]] <- as.numeric(metrics_df[[metric]])
  
  # Update labels for readability
  metrics_df$treatment <- factor(metrics_df$treatment,
                                 levels = c("original", "original_boxed", "segment", "segment_bodyPart"),
                                 labels = c("original", "original\nboxed", "segment", "segmented\nbody part"))
  
  metrics_df$assess_area <- factor(metrics_df$assess_area,
                                   levels = c("hindleg cleanliness", "hindquarter cleanliness", "udder cleanliness"),
                                   labels = c("hindleg", "hindquarter", "udder"))
  
  # Plot with updated aesthetics
  plot <- ggplot(metrics_df, aes(x=assess_area, y=treatment, fill=get(metric))) +
    geom_tile(color = "white", size = 0.5) +  # Adding white borders for clearer separation
    geom_text(aes(label = sprintf("%.2f", get(metric))), vjust = 0.5, hjust = 0.5, color = "white", size = 6) +  # Center values in each cell
    scale_fill_gradient(low = "lavender", high = "mediumorchid3", guide = "none") +  # Apply new color gradient
    labs(title = paste("Heatmap of", metric), x = "Assessment Area", y = "Treatment") +
    theme_classic(base_size = 25) +  # Apply classic theme with larger base text size
    theme(
      axis.title = element_text(size = 28),  # Larger axis titles
      plot.title = element_text(size = 30)   # Larger plot title
    )
  
  return(plot)
}