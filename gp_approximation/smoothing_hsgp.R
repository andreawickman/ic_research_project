require(dplyr)
require(ggplot2)
require(rstan)
SEED <- 50
set1 <- RColorBrewer::brewer.pal(7, "Set1")

indir <- './saving_lives/data/derived'
filename <- file.path(indir,'imputed_phylo_data.csv')
lrtt_data <- read.csv(filename)

sample_id <- "AID4174-fq1"
sample_data <- lrtt_data %>% filter(RENAME_ID == sample_id & !is.na(normalised.largest.rtt) & normalised.largest.rtt>0)

# try fitting to all TSIs >2 years -> very slow
#sample_data <- lrtt_data %>% filter(sci_years_cat == ">=2 years" & !is.na(normalised.largest.rtt) & normalised.largest.rtt>0)

# Extract x and y variables
x <- sample_data$xcoord
y <- sample_data$normalised.largest.rtt

# Plot one sample
sample_data %>%
  ggplot(aes(x=x,y=y))+
  geom_point()+
  labs(x="Genome Coordinate", y="LRTT") +
  geom_smooth()

standata_sample <- list(x = x,
                        y = y,
                        N = length(x),
                        #G = length(unique(sample_data$AID)),
                        #group = as.numeric(factor(sample_data$AID)),
                        c_f = 1.5,  # factor c of basis functions for GP for f1
                        M_f = 30   # number of basis functions for GP for f1
)

# Define the Stan model file path
stan_file <- "./saving_lives/hsgp_gamma.stan"

# Compile the Stan model
#stan_model <- cmdstanr::cmdstan_model(file1,force_recompile = TRUE)
init = list(
  intercept = 0,
  beta_f = rep(0.1, standata_sample$M_f),
  lengthscale_f = 0.1,
  sigma_f = 0.5,
  beta = 1
)
rstan_options(auto_write = FALSE)
fit <- stan(
  file = stan_file,
  data = standata_sample,
  iter = 2000,  # Adjust the number of iterations as needed
  warmup = 500,  # Adjust the warmup iterations as needed
  chains = 4,
  seed = SEED,
  control = list(adapt_delta = 0.9)
)

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
combined_data <- cbind(sample_data, posterior_summary)

combined_data %>%
  ggplot(aes(x = xcoord, y = normalised.largest.rtt)) +
  geom_point() +  # Plot original data
  geom_line(aes(y = median), color = set1[1]) +  # Plot median predictions
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.5, fill = set1[1]) +  # 95% credible interval
  labs(x = "Genome Coordinate", y = "LRTT") +
  theme_minimal()
