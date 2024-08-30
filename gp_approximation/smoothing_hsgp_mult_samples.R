set.seed(2509292)
sample_ids <- sample(unique(lrtt_data$RENAME_ID), 4)
#sample_ids <- c("AID4174-fq1", "AID0053-fq1", "AID6734-fq1", "AID3361-fq1", "AID6082-fq1", "")  # Add more sample IDs here

# Prepare an empty list to store the combined data
combined_data_list <- list()

# Loop over each sample and fit the model
for (sample_id in sample_ids) {
  
  sample_data <- lrtt_data %>% 
    filter(RENAME_ID == sample_id & !is.na(normalised.largest.rtt) & normalised.largest.rtt > 0)
  
  # Extract x and y variables
  x <- sample_data$xcoord
  y <- sample_data$normalised.largest.rtt
  
  # Prepare the data for Stan
  standata_sample <- list(
    x = x,
    y = y,
    N = length(x),
    c_f = 1.5,  # factor c of basis functions for GP for f1
    M_f = 30    # number of basis functions for GP for f1
  )
  
  # Fit the model using Stan
  fit <- stan(
    file = stan_file,
    data = standata_sample,
    iter = 2000,  # Adjust the number of iterations as needed
    warmup = 500,  # Adjust the warmup iterations as needed
    chains = 4,
    seed = SEED,
    control = list(adapt_delta = 0.9)
  )
  
  # Extract posterior samples and compute summary statistics
  posterior_samples <- extract(fit)
  y_pred_samples <- posterior_samples$y_pred
  
  # Summarize the posterior draws for each observation (column)
  posterior_summary <- apply(y_pred_samples, 2, function(x) {
    c(median = median(x),
      q2.5 = quantile(x, 0.025),
      q97.5 = quantile(x, 0.975))
  })
  
  posterior_summary <- as.data.frame(t(posterior_summary))
  colnames(posterior_summary) <- c("median", "lower", "upper")
  
  # Combine the posterior summary with the original data
  combined_data <- cbind(sample_data, posterior_summary)
  combined_data$sample_id <- sample_id  # Add sample ID column
  
  # Store the result in the list
  combined_data_list[[sample_id]] <- combined_data
}

# Combine all the data into one dataframe
all_combined_data <- do.call(rbind, combined_data_list)
# Plot all samples together with facets
all_combined_data %>%
  ggplot(aes(x = xcoord, y = normalised.largest.rtt, color = sample_id)) +
  geom_point() +  # Plot original data
  geom_line(aes(y = median)) +  # Plot median predictions
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = sample_id), alpha = 0.3) +  # 95% credible interval
  labs(x = "Genome Coordinate", y = "LRTT") +
  facet_wrap(~ sample_id, scales = "free_y") +  # Facet by sample
  theme_minimal() +
  theme(legend.position = "none")