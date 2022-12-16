// modified from https://github.com/stan-dev/example-models/blob/master/misc/cluster/lda/corr-lda.stan
// apply trick to get gqs() to work with transformed parameters from https://discourse.mc-stan.org/t/k-fold-validation-for-hierarchical-model-in-rstan/23173/6
data {
  int<lower=2> K; // num topics
  int<lower=2> V; // num words
  int<lower=1> M; // num docs
  int<lower=1> N; // total word instances
  int<lower=1,upper=V> w[N];    // word n
  int<lower=1,upper=M> doc[N];  // doc ID for word n
  vector<lower=0>[V] beta; // word prior
}
parameters {
  vector[K] mu; // topic mean
  // corr_matrix[K] Omega; // correlation matrix
  cholesky_factor_corr[K] Lcorr; // cholesky factor of correlation matrix
  vector<lower=0>[K] sigma; // scales
  vector[K] eta[M]; // logit topic dist for doc m
  simplex[V] phi[K]; // word dist for topic k
}
model {
}
generated quantities {
  simplex[K] theta[M]; // simplex topic dist for doc m
  matrix[K,K] Omega;   // correlation matrix
  cov_matrix[K] Sigma; // covariance matrix
  vector[N] log_lik;
  
  for (m in 1:M) {
    theta[m] = softmax(eta[m]);
  }
  Omega = multiply_lower_tri_self_transpose(Lcorr);
  Sigma = quad_form_diag(Omega, sigma); 
  
  for (n in 1:N) {
    real gamma[K];
    for (k in 1:K)
      gamma[k] = log(theta[doc[n], k]) + log(phi[k, w[n]]);
    log_lik[n] = log_sum_exp(gamma);  // likelihood;
  }
}
