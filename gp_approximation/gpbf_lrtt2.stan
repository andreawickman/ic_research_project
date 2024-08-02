functions {
  vector diagSPD_EQ(real alpha, real rho, real L, int M) {
    return sqrt((alpha^2) * sqrt(2*pi()) * rho * exp(-0.5 * (rho*pi() / (2 * L))^2 * linspaced_vector(M, 1, M)^2));
  }

  matrix PHI_EQ(int N, int M, real L, vector x) {
    return sin(diag_post_multiply(rep_matrix(pi() / (2*L) * (x+L), M), linspaced_vector(M, 1, M))) / sqrt(L);
  }
}
data {
  int<lower=1> N;           // number of observations
  vector[N] x;              // univariate covariate
  vector<lower=0>[N] y;     // target variable (must be > 0)
  
  real<lower=0> c_f;        // factor c to determine the boundary value L
  int<lower=1> M_f;         // number of basis functions
  real<lower=0> c_g;        // factor c to determine the boundary value L
  int<lower=1> M_g;         // number of basis functions
}
transformed data {
  // Normalize data
  real xmean = mean(x);
  real ymean = mean(y);     // Mean of y
  real xsd = sd(x);
  real ysd = sd(y);         // Standard deviation of y
  vector[N] xn = (x - xmean) / xsd;
  vector[N] yn = (y - ymean) / ysd; // Standardize y
  
  // Basis functions for f
  real L_f = c_f * max(xn);
  matrix[N, M_f] PHI_f = PHI_EQ(N, M_f, L_f, xn);
  
  // Basis functions for g
  real L_g = c_g * max(xn);
  matrix[N, M_g] PHI_g = PHI_EQ(N, M_g, L_g, xn);
}
parameters {
  real<lower=0.01> intercept;   // Ensure intercept is positive
  vector[M_f] beta_f;           // Basis functions coefficients
  vector[M_g] beta_g;           // Basis functions coefficients
  real<lower=0.01> lengthscale_f; // Ensure positive lengthscale
  real<lower=0.01> sigma_f;     // Ensure positive scale
  real<lower=0.01> lengthscale_g; // Ensure positive lengthscale
  real<lower=0.01> sigma_g;     // Ensure positive scale
}
model {
  // Spectral densities for f and g
  vector[M_f] diagSPD_f = diagSPD_EQ(sigma_f, lengthscale_f, L_f, M_f);
  vector[M_g] diagSPD_g = diagSPD_EQ(sigma_g, lengthscale_g, L_g, M_g);

  // Priors
  intercept ~ normal(1, 0.5);  // Adjusted to ensure reasonable positive values
  beta_f ~ normal(0, 0.25);     // Narrower prior for coefficients
  beta_g ~ normal(0, 0.25);     // Narrower prior for coefficients
  lengthscale_f ~ normal(0.5, 0.5); // Adjusted for reasonable positive values
  lengthscale_g ~ normal(0.5, 0.5); // Adjusted for reasonable positive values
  sigma_f ~ normal(0.5, 0.5);  // Adjusted for reasonable positive values
  sigma_g ~ normal(0.5, 0.5);  // Adjusted for reasonable positive values
  
  // Likelihood using gamma distribution
  y ~ gamma(intercept + PHI_f * (diagSPD_f .* beta_f) + 0.01, //ensure shape parameter is positive 
            exp(PHI_g * (diagSPD_g .* beta_g))); //scale parameter
}
generated quantities {
  vector[N] f;
  vector[N] sigma;
  
  {
    // Spectral densities
    vector[M_f] diagSPD_f = diagSPD_EQ(sigma_f, lengthscale_f, L_f, M_f);
    vector[M_g] diagSPD_g = diagSPD_EQ(sigma_g, lengthscale_g, L_g, M_g);
    
    // Function scaled back to the original scale
    f = (intercept + PHI_f * (diagSPD_f .* beta_f)) * ysd + ymean;
    sigma = exp(PHI_g * (diagSPD_g .* beta_g)) * ysd;
  }
}
