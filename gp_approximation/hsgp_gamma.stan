functions {
  vector diagSPD_EQ(real alpha, real rho, real L, int M) {
    return sqrt((alpha^2) * sqrt(2*pi()) * rho * exp(-0.5*(rho*pi()/(2*L))^2 * linspaced_vector(M, 1, M)^2));
  }

  matrix PHI_EQ(int N, int M, real L, vector x) {
    return sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
  }
}
data {
  int<lower=1> N;         // number of observations
  vector[N] x;            // univariate covariate
  vector<lower=0>[N] y;   // target variable (must be > 0)

  real<lower=0> c_f;      // factor c to determine the boundary value L for f
  int<lower=1> M_f;       // number of basis functions for f
}

transformed data {
  // Normalize data
  real xmean = mean(x);
  real ymean = mean(y);
  real xsd = sd(x);
  real ysd = sd(y);
  vector[N] xn = (x - xmean) / xsd;
  //vector[N] yn = (y - ymean) / ysd;

  // Basis functions for f
  real L_f = c_f * max(xn);
  matrix[N, M_f] PHI_f = PHI_EQ(N, M_f, L_f, xn);
}

parameters {
  real intercept;              // Intercept for the mean function
  vector[M_f] beta_f;          // Coefficients for basis functions for f
  real<lower=0> lengthscale_f; // Lengthscale for f
  real<lower=0> sigma_f;       // Scale for f
  real logbeta;          // Inverse scale parameter for the Gamma distribution (constrained to positive)
}

transformed parameters {
  vector[N] mu; // Shape parameter (must be positive)
  real beta;
  vector[N] alpha;

  // Spectral density for f
  vector[M_f] diagSPD_f = diagSPD_EQ(sigma_f, lengthscale_f, L_f, M_f);

  // Calculate mu and ensure positivity with exp()
  mu = exp(intercept + PHI_f * (diagSPD_f .* beta_f));
  beta = exp(-logbeta);
  alpha = mu * beta;
}

model {

  // Priors
  intercept ~ normal(0, 1);
  beta_f ~ normal(0, 1); 
  lengthscale_f ~ inv_gamma(5, 2); //this is 5/4 try making beta smaller beta/alpha-1
  sigma_f ~ normal(0, 0.15);
  logbeta ~ normal(0, 5); // Prior on inverse scale, adjusted to avoid extreme values

  // Model (Gamma likelihood)
  y ~ gamma(mu, beta);
}

generated quantities {
  vector[N] y_pred;

  // Generate predictions
  for (n in 1:N) {
    y_pred[n] = gamma_rng(mu[n], beta);
  }
}
//fitting independent gp for every sample - computationally expensive 
//fitting hierarchical - lenghtscale_f and sigma_f estimate for individual, but their prior shares information across 
