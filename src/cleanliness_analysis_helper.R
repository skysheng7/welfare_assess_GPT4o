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
    accurate_predict_pct <- (accurate_predictions / total_predictions) * 100
    
    # Combine the current row with frequencies and accuracy percentage into a new data frame
    current_result <- cbind(unique_images[i, ], predict_score_0_freq = score_0_freq, predict_score_2_freq = score_2_freq, accurate_predict_pct = accurate_predict_pct)
    
    # Append the result to the list
    results_list[[i]] <- current_result
  }
  
  # Combine all individual data frames into one data frame
  final_results <- do.call(rbind, results_list)
  return(final_results)
}