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

calculate_summary_metrics <- function(subset_data) {
  summary_subset <- summarize_scores(subset_data)
  summary_subset$inaccurate_pct <- 1 - summary_subset$accurate_predict_pct
  
  accuracy <- mean(summary_subset$accurate_predict_pct)
  prec_rec_metrics <- weighted_precision_recall(summary_subset)
  
  return(c(as.numeric(accuracy), 
           as.numeric(prec_rec_metrics$precision), 
           as.numeric(prec_rec_metrics$recall)))
}

calculate_metrics <- function(data) {
  unique_areas = unique(data$assess_area)
  unique_treatments = unique(data$treatment)
  unique_combinations <- unique(data[c("assess_area", "treatment")])
  metrics_df <- data.frame(assess_area=character(), treatment=character(), 
                           Accuracy=numeric(), Precision=numeric(), Recall=numeric(), 
                           stringsAsFactors = FALSE)
  
  # Process each unique combination
  for (i in seq(nrow(unique_combinations))) {
    subset_data <- data[data$assess_area == unique_combinations$assess_area[i] & 
                          data$treatment == unique_combinations$treatment[i],]
    metrics <- calculate_summary_metrics(subset_data)
    metrics_df[nrow(metrics_df) + 1, ] <- c(unique_combinations$assess_area[i], 
                                            unique_combinations$treatment[i], metrics)
  }
  
  # Overall metrics for each assess_area across all treatments
  for (area in unique_areas) {
    subset_data <- data[data$assess_area == area,]
    metrics <- calculate_summary_metrics(subset_data)
    metrics_df[nrow(metrics_df) + 1, ] <- c(area, "overall", metrics)
  }
  
  # Overall metrics for each treatment across all assessment areas
  for (treatment in unique_treatments) {
    subset_data <- data[data$treatment == treatment,]
    metrics <- calculate_summary_metrics(subset_data)
    metrics_df[nrow(metrics_df) + 1, ] <- c("overall", treatment, metrics)
  }
  
  # Overall metrics across all treatments and assess_area
  metrics <- calculate_summary_metrics(data)
  metrics_df[nrow(metrics_df) + 1, ] <- c("overall", "overall", metrics)
  
  return(metrics_df)
}


plot_heatmap <- function(metrics_df, metric) {
  # Ensure the metric column is treated as numeric
  metrics_df[[metric]] <- as.numeric(metrics_df[[metric]])
  
  # Update labels for readability and reorder treatments
  metrics_df$treatment <- factor(metrics_df$treatment,
                                 levels = c("overall", "segment_bodyPart", "segment", "original_boxed", "original"),
                                 labels = c("overall", "segmented\nbody part", "segment", "original\nboxed", "original"))
  
  metrics_df$assess_area <- factor(metrics_df$assess_area,
                                   levels = c("hindleg cleanliness", "hindquarter cleanliness", "udder cleanliness", "overall"),
                                   labels = c("hindleg", "hindquarter", "udder", "overall"))
  
  # Plot with updated aesthetics
  plot <- ggplot(metrics_df, aes(x=assess_area, y=treatment, fill=get(metric))) +
    geom_tile(color = "white", size = 0.5) +  # Adding white borders for clearer separation
    geom_text(aes(label = sprintf("%.2f", get(metric))), vjust = 0.5, hjust = 0.5, 
              color = "black", size = 8, fontface = "bold") +  # Bold, black, and bigger text in each cell
    scale_fill_gradient(low = "white", high = "blue", 
                        guide = "none", limits = c(0.35, 1.05)) +  # Apply new color gradient with fixed limits
    #labs(title = metric, x = "Assessment Area", y = "Treatment Type") +
    labs(title = metric, x = NULL, y = NULL) +
    theme_classic(base_size = 27) +  # Apply classic theme with larger base text size
    theme(
      axis.text.y = element_text(color = "black"),  # Ensure y-axis labels are visible
      axis.ticks = element_blank(),  # Remove axis ticks
      axis.ticks.length = unit(0, "points"),  # Ensure no tick marks
      axis.line = element_blank(),  # Remove the axis lines
      plot.title = element_text(size = 30, hjust = 0.5)  # Center the plot title
    ) +
    scale_x_discrete(position = "top")  # Move x-axis labels to the top
  
  return(plot)
}