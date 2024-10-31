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
  double lp_diff = log_prior_proposed - log_prior_current;
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
  
  si.ls_sigma_e = calculate_adaptive_ls(si.ac_sigma_e, ac_count, si.ls_sigma_e, ls_incr);
  si.ls_sigma_u = calculate_adaptive_ls(si.ac_sigma_u, ac_count, si.ls_sigma_u, ls_incr);
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
  for(size_t i=0;i<di.n;i++) {
    si.u_sample[i] = gen.normal(0, si.sigma_u);
  }
}