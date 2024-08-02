functions {
  vector diagSPD_EQ(real alpha, real rho, real L, int M) {
    return sqrt((alpha^2) * sqrt(2*pi()) * rho * exp(-0.5 * (rho*pi() / (2 * L))^2 * linspaced_vector(M, 1, M)^2));
  }

  vector diagSPD_periodic(real alpha, real rho, int M) {
    real a = 1/rho^2;
    int one_to_M[M];
    for (m in 1:M) one_to_M[m] = m;
    vector[M] q = sqrt(alpha^2 * 2 / exp(a) * to_vector(modified_bessel_first_kind(one_to_M, a)));
    return append_row(q, q);
  }

  matrix PHI_EQ(int N, int M, real L, vector x) {
    return sin(diag_post_multiply(rep_matrix(pi() / (2*L) * (x+L), M), linspaced_vector(M, 1, M))) / sqrt(L);
  }

  matrix PHI_periodic(int N, int M, real w0, vector x) {
    matrix[N, M] mw0x = diag_post_multiply(rep_matrix(w0*x, M), linspaced_vector(M, 1, M));
    return append_col(cos(mw0x), sin(mw0x));
  }
}

data {
  int<lower=1> N;      // number of observations
  vector[N] x;         // univariate covariate
  vector<lower=0>[N] y;         // target variable
  
  real<lower=0> c_f;   // factor c to determine the boundary value L
  int<lower=1> M_f;    // number of basis functions
  real<lower=0> c_g;   // factor c to determine the boundary value L
  int<lower=1> M_g;    // number of basis functions
  real w0;             // frequency for periodic kernel
}

transformed data {
  // Normalize data
  real xmean = mean(x);
  real ymean = mean(y);
  real xsd = sd(x);
  real ysd = sd(y);
  vector[N] xn = (x - xmean) / xsd;
  vector[N] yn = (y - ymean) / ysd;
  
  // Basis functions for f
  real L_f = c_f * max(xn);
  matrix[N, M_f] PHI_f = PHI_EQ(N, M_f, L_f, xn);
  
  // Basis functions for g
  real L_g = c_g * max(xn);
  matrix[N, M_g] PHI_g = PHI_periodic(N, M_g, w0, xn);
}

parameters {
  real intercept;
  vector[M_f] beta_f;       // Basis functions coefficients
  vector[M_g] beta_g;       // Basis functions coefficients
  real<lower=0.01> lengthscale_f; // Ensure positive lengthscale
  real<lower=0.01> sigma_f;       // Ensure positive scale
  real<lower=0.01> lengthscale_g; // Ensure positive lengthscale
  real<lower=0.01> sigma_g;       // Ensure positive scale
  real<lower=0> phi;              // Ensure positive phi
}

model {
  // Spectral densities for f and g
  vector[M_f] diagSPD_f = diagSPD_EQ(sigma_f, lengthscale_f, L_f, M_f);
  vector[M_g] diagSPD_g = diagSPD_periodic(sigma_g, lengthscale_g, M_g);

  // Priors
  intercept ~ normal(0, 1);
  beta_f ~ normal(0, 1);
  beta_g ~ normal(0, 1);
  lengthscale_f ~ normal(1, 1);
  lengthscale_g ~ normal(1, 1);
  sigma_f ~ normal(1, .5);
  sigma_g ~ normal(1, .5);
  phi ~ normal(0, 1);    // Prior for phi
  
  // Likelihood
  y ~ gamma(exp(intercept + PHI_f * (diagSPD_f .* beta_f)),
            exp(PHI_g * (diagSPD_g .* beta_g)) / phi);
}

generated quantities {
  vector[N] mu;
  vector[N] sigma;
  vector[N] y_pred;
  
  {
    // Spectral densities
    vector[M_f] diagSPD_f = diagSPD_EQ(sigma_f, lengthscale_f, L_f, M_f);
    vector[M_g] diagSPD_g = diagSPD_periodic(sigma_g, lengthscale_g, M_g);
    
    // Function scaled back to the original scale
    mu = exp(intercept + PHI_f * (diagSPD_f .* beta_f)) * ysd + ymean;
    sigma = exp(PHI_g * (diagSPD_g .* beta_g)) * ysd;
    for (n in 1:N) {
      y_pred[n] = gamma_rng(mu[n], sigma[n] / phi);
    }
  }
}