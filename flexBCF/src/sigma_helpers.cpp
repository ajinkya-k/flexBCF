#include "sigma_helpers.h"


// generates proposal for a new sigma
double propose_sigma(double sigma_current, double ls_proposal, RNG& gen) {
  double delta = sqrt(exp(2*ls_proposal));
  double log_proposal = log(sigma_current) + gen.normal(0., 1.) * delta;
  double proposal = exp(log_proposal);
  return(proposal);
}


void calculate_sigma2_i(double sigma_e, double sigma_u, int n, double* wts, double* return_loc) {
  // precalculate squares rather than calculating inside loop
  // Rcpp::Rcout << "sigma_e = " << sigma_e << ", sigma_u = " << sigma_u << ", n = " << n << std::endl;
  double v_y = sigma_e * sigma_e;
  double v_u = sigma_u * sigma_u;

  double* test = new double[n];
  for (int i=0; i < n; i++) test[i] = wts[i] * 2;

  for (int i=0; i < n; i++){
    return_loc[i] = v_y/wts[i] + v_u;
  } 
}


double calculate_lp_diff(sigma_info &si, data_info &di, double log_prior_current, double log_prior_proposed) {
  // Log likelihood requires two different sums: sum of the log of sigma_i^2, and sum of resid/sigma_i^2
  double sum_log_sig2_i_current     = 0;
  double sum_r_over_sig2_i_current  = 0;
  double sum_log_sig2_i_proposed    = 0;
  double sum_r_over_sig2_i_proposed = 0;

  double r, r2, sigma2_current, sigma2_proposed;
  for (size_t i=0;i<di.n;i++) {
    r = di.rp[i];
    r2 = r*r;
    sigma2_current  = di.var_i[i];
    sigma2_proposed = si.prop_var_i[i];
    
    sum_log_sig2_i_current  += log(sigma2_current);
    sum_log_sig2_i_proposed += log(sigma2_proposed);

    sum_r_over_sig2_i_current  += r2/sigma2_current;
    sum_r_over_sig2_i_proposed += r2/sigma2_proposed;
  }
  // Now compose the log posteriors: log prior + log likelihood
  double lp_current  = log_prior_current  -0.5 * sum_log_sig2_i_current  - 0.5 * sum_r_over_sig2_i_current;
  double lp_proposed = log_prior_proposed -0.5 * sum_log_sig2_i_proposed - 0.5 * sum_r_over_sig2_i_proposed;

  double lp_diff = lp_proposed - lp_current;
  return(lp_diff);
}

double calculate_adaptive_ls(int accepted, double target, double log_sigma, double increment) {
  if (accepted < target) {
    log_sigma += -increment;
  } else if (accepted > target) {
    log_sigma += increment;
  }
  return(log_sigma);
}

void update_adaptive_ls(sigma_info& si, size_t iter, int batch_size, double ac_target) {
  // Convert from target acceptance % to target # accepted iters
  double ac_count = ac_target * (iter+1);
  // We will incrememnt the log_sigma terms by 1/# batches completed
  double ls_incr = batch_size / (iter + 1.);
  
  si.ls_sigma_e   = calculate_adaptive_ls(si.ac_sigma_e,   ac_count, si.ls_sigma_e,   ls_incr);
  si.ls_sigma_u   = calculate_adaptive_ls(si.ac_sigma_u,   ac_count, si.ls_sigma_u,   ls_incr);
  si.ls_mu_scale  = calculate_adaptive_ls(si.ac_mu_scale,  ac_count, si.ls_mu_scale,  ls_incr);
  si.ls_tau_scale = calculate_adaptive_ls(si.ac_tau_scale, ac_count, si.ls_tau_scale, ls_incr);
}


void update_sigma_e(sigma_info &si, data_info &di, double nu, double lambda, double* wts, RNG& gen) {
  // Proposal is an adaptive MH draw, scaled by ls_sigma_y
  double s_e = si.sigma_e;
  double s_u = si.sigma_u;
  double proposal = propose_sigma(s_e, si.ls_sigma_e, gen);
  // sigma_info &si, data_info &di, double* wts, double* return_loc
  calculate_sigma2_i(proposal, s_u, di.n, wts, si.prop_var_i);   

  double log_prior_current  = - (nu/2 + 1) * log(s_e*s_e) - nu*lambda / (2*s_e*s_e);
  double log_prior_proposed = - (nu/2 + 1) * log(proposal * proposal)   - nu*lambda / (2*proposal  *proposal);
  double lp_diff = calculate_lp_diff(si, di, log_prior_current, log_prior_proposed);
  double log_ratio = lp_diff + log(proposal) - log(s_e);

  //Accept or reject
  double cut = gen.uniform();
  if (log(cut) < log_ratio) {
    // gi.logger.log("Accepting proposed sigma_y " + std::to_string(proposal));
    si.sigma_e = proposal;
    si.ac_sigma_e += 1;
    for (size_t i; i<di.n; ++i) {
      di.var_i[i] = si.prop_var_i[i];
    }
  } else {
    //gi.logger.log("Rejecting proposed sigma_y " + std::to_string(proposal));
    // Rcpp::Rcout << "Rejecting proposed sigma_y " << proposal << std::endl;
  }
}

void update_sigma_u(sigma_info& si, data_info &di, double hyperprior, double* wts, RNG &gen) {
  // Proposal is an adaptive MH draw, scaled by ls_sigma_u
  double proposal = propose_sigma(si.sigma_u, si.ls_sigma_u, gen);
  calculate_sigma2_i(si.sigma_e, proposal, di.n, wts, si.prop_var_i);

  double log_prior_current =  -si.sigma_u*si.sigma_u / (2 * hyperprior * hyperprior);
  double log_prior_proposed = -proposal  *proposal   / (2 * hyperprior * hyperprior);
  double lp_diff = calculate_lp_diff(si, di, log_prior_current, log_prior_proposed);
  double log_ratio = lp_diff + log(proposal) - log(si.sigma_u);

  //Accept or reject
  double cut = gen.uniform();
  if (log(cut) < log_ratio) {
    // gi.logger.log("Accepting proposed sigma_u " + std::to_string(proposal));
    si.sigma_u = proposal;
    si.ac_sigma_u += 1;
    for (size_t i; i<di.n; ++i) {
      di.var_i[i] = si.prop_var_i[i];
    }
  } else {
    // Rcpp::Rcout << "Rejecting proposed sigma_u " << proposal << std::endl;
    // gi.logger.log("Rejecting proposed sigma_u " + std::to_string(proposal));
  }
}


void draw_u(sigma_info &si, data_info& di, double* w, RNG &gen) {
  double v_y = si.sigma_e * si.sigma_e;
  double v_u = si.sigma_u * si.sigma_u;
  double prior_prec = 1/v_u;
  double r, data_prec, post_prec, post_sd, post_mean;
  for(size_t i=0;i<di.n;i++) {
    data_prec = w[i] / (v_y);
    post_prec = prior_prec + data_prec;
    post_sd = sqrt(1/post_prec);
    post_mean = data_prec * di.rp[i] / post_prec;

    si.u_sample[i] = gen.normal(post_mean, post_sd);
  }
}

void update_mu_scale(sigma_info &si, data_info& di,
                      double* allfit, double* allfit_proposed, 
                      double* mu_train,
                      Rcpp::NumericVector Y_train, RNG& gen) {
    double proposal = propose_sigma(di.mu_scale, si.ls_mu_scale, gen);
    double scale_ratio = proposal/di.mu_scale;
    double scale_diff = proposal - di.mu_scale;
    for (int k=0; k<di.n; ++k) {
      allfit_proposed[k] = allfit[k] + mu_train[k] * scale_diff;
    }
    
    // Implicitly the denominator is 2*hyperprior variance, but hyperprior var is 1
    double log_prior_current =  -di.mu_scale * di.mu_scale      / (2);
    double log_prior_proposed = -proposal    * proposal   / (2);
    
    double lp_diff = calculate_lp_diff_forscales(allfit, allfit_proposed, log_prior_current, log_prior_proposed, di.n, di.var_i, Y_train);
    double log_ratio = lp_diff + log(proposal) - log(di.mu_scale);

    //Accept or reject
    double cut = gen.uniform();

    Rcpp::Rcout << "muscale.curr_ms " << di.mu_scale << std::endl;
    Rcpp::Rcout << "muscale.curr_af " << allfit[di.n-1] << std::endl;
    Rcpp::Rcout << "muscale.curr_mu " << mu_train[di.n-1] << std::endl;
    Rcpp::Rcout << "muscale.prop_ms " << proposal << std::endl;
    Rcpp::Rcout << "muscale.prop_af " << allfit_proposed[di.n-1] << std::endl;
    Rcpp::Rcout << "muscale.prop_mu " << mu_train[di.n-1] * scale_ratio << std::endl;
    Rcpp::Rcout << "muscale.curr_lp " << log_prior_current << std::endl;
    Rcpp::Rcout << "muscale.prop_lp " << log_prior_proposed << std::endl;
    Rcpp::Rcout << "muscale.lpdiff " << lp_diff << std::endl;
    Rcpp::Rcout << "muscale.lograt " << log_ratio << std::endl;
    Rcpp::Rcout << "muscale.cut " << cut << std::endl;

    if (log(cut) < log_ratio) {
      di.mu_scale = proposal;
      si.ac_mu_scale += 1;
      for(int k=0; k<di.n; ++k) {
        allfit[k]      = allfit_proposed[k];
        mu_train[k]    = mu_train[k] * scale_ratio;
      }
    } else {
      // Nothing, we rejected
    }
}

void update_tau_scale(sigma_info &si, data_info& di,
                      double* allfit, double* allfit_proposed, 
                      double* tau_train,
                      Rcpp::NumericVector Y_train, RNG& gen) {
    double proposal = propose_sigma(di.tau_scale, si.ls_tau_scale, gen);
    double scale_ratio = proposal/di.tau_scale;
    double scale_diff = proposal - di.tau_scale;
    for (int k=0; k<di.n; ++k) {
      allfit_proposed[k] = allfit[k] + tau_train[k] * scale_diff;
    }
    
    // Implicitly the denominator is 2*hyperprior variance, but hyperprior var is 1
    double log_prior_current =  -di.tau_scale * di.tau_scale      / (2);
    double log_prior_proposed = -proposal    * proposal   / (2);
    
    double lp_diff = calculate_lp_diff_forscales(allfit, allfit_proposed, log_prior_current, log_prior_proposed, di.n, di.var_i, Y_train);
    double log_ratio = lp_diff + log(proposal) - log(di.tau_scale);

    //Accept or reject
    double cut = gen.uniform();

    Rcpp::Rcout << "tauscale.curr_ts " << di.tau_scale << std::endl;
    Rcpp::Rcout << "tauscale.curr_af " << allfit[di.n-1] << std::endl;
    Rcpp::Rcout << "tauscale.curr_tau " << tau_train[di.n-1] << std::endl;
    Rcpp::Rcout << "tauscale.prop_ts " << proposal << std::endl;
    Rcpp::Rcout << "tauscale.prop_af " << allfit_proposed[di.n-1] << std::endl;
    Rcpp::Rcout << "tauscale.prop_tau " << tau_train[di.n-1] * scale_ratio << std::endl;
    Rcpp::Rcout << "tauscale.curr_lp " << log_prior_current << std::endl;
    Rcpp::Rcout << "tauscale.prop_lp " << log_prior_proposed << std::endl;
    Rcpp::Rcout << "tauscale.lpdiff " << lp_diff << std::endl;
    Rcpp::Rcout << "tauscale.lograt " << log_ratio << std::endl;
    Rcpp::Rcout << "tauscale.cut " << cut << std::endl;

    if (log(cut) < log_ratio) {
      di.tau_scale = proposal;
      si.ac_tau_scale += 1;
      for(int k=0; k<di.n; ++k) {
        allfit[k]      = allfit_proposed[k];
        tau_train[k]    = tau_train[k] * scale_ratio;
      }
    } else {
      // Nothing, we rejected
    }
}

double calculate_lp_diff_forscales(double* allfit, double* allfit_proposed, 
                                    double log_prior_current, double log_prior_proposed,
                                    int n_train, double* var_i, Rcpp::NumericVector Y_train) {
  // Log likelihood requires two different sums: sum of the log of sigma_i^2, and sum of resid/sigma_i^2
  double sum_r_over_sig2_i_current  = 0;
  double sum_r_over_sig2_i_proposed = 0;

  double r_current, r2_current, r_proposed, r2_proposed;
  for (int i=0; i<n_train; i++) {
    r_current   = Y_train[i] - allfit[i];
    r2_current  = r_current*r_current;
    r_proposed  = Y_train[i] - allfit_proposed[i];
    r2_proposed = r_proposed*r_proposed;

    sum_r_over_sig2_i_current  += r2_current  / var_i[i];
    sum_r_over_sig2_i_proposed += r2_proposed / var_i[i];
  }
  // Now compose the log posteriors: log prior + log likelihood
  // thje logposterior also includes sum(log*sigma2_i), but because that is the same for both current and proposal, we can ignore in the diff since it falls out as a proportionality constant
  double lp_current  = log_prior_current  - 0.5 * sum_r_over_sig2_i_current;
  double lp_proposed = log_prior_proposed - 0.5 * sum_r_over_sig2_i_proposed;

  double lp_diff = lp_proposed - lp_current;
  return(lp_diff);
}