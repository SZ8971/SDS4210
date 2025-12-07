data {
  // --- Training Data ---
  int<lower=0> N_train;
  array[N_train] int home_team_train;
  array[N_train] int away_team_train;
  array[N_train] int home_win_train; // 0 or 1

  // --- Testing Data (For y_new_rep) ---
  int<lower=0> N_test;
  array[N_test] int home_team_test;
  array[N_test] int away_team_test;
  
  // --- Dimensions ---
  int<lower=0> N_teams; 
}

parameters {
  real alpha;              // Home Field Advantage
  vector[N_teams] theta;   // Team Strength
  real<lower=0> sigma;     // Variation in team strength
}

model {
  // Priors
  theta ~ normal(0, sigma);
  alpha ~ normal(0, 1);
  sigma ~ cauchy(0, 2);

  // Likelihood (Training Data Only)
  for (n in 1:N_train) {
    home_win_train[n] ~ bernoulli_logit(alpha + theta[home_team_train[n]] - theta[away_team_train[n]]);
  }
}

generated quantities {
  vector[N_test] y_new_rep; // Probability (0.0 to 1.0)
  int y_new_outcome[N_test]; // Simulated Outcome (0 or 1)
  
  for (n in 1:N_test) {
    real log_odds = alpha + theta[home_team_test[n]] - theta[away_team_test[n]];
    y_new_rep[n] = inv_logit(log_odds);
    y_new_outcome[n] = bernoulli_logit_rng(log_odds); // The literal coin flip
  }
}