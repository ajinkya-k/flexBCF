#ifndef GUARD_sigma_helpers_het_h
#define GUARD_sigma_helpers_het_h

#include "helpers.h"
#include "rng.h"

// class holding data dimensions and pointers to the covariate data
class sigma_info{
public:
  double sigma_e;
  double sigma_u;
  double ls_sigma_e;
  double ls_sigma_u;
  int ac_sigma_e;
  int ac_sigma_u;

  double ls_mu_scale;
  double ls_tau_scale;
  int ac_mu_scale;
  int ac_tau_scale;

  double* prop_var_i;
  double* u_sample;

  sigma_info(){
    sigma_e = 1.0;
    sigma_u = 1.0;
    ac_sigma_e = 0;
    ac_sigma_u = 0;
  }
};


double propose_sigma(double sigma_current, double ls_proposal, RNG& gen);
void calculate_sigma2_i(double sigma_e, double sigma_u, int n, double* wts, double* return_loc);
double calculate_lp_diff(sigma_info &si, data_info &di, double log_prior_current, double log_prior_proposed, bool prior_only);
double calculate_adaptive_ls(int accepted, double target, double log_sigma, double increment);
void update_adaptive_ls(sigma_info& si, size_t iter, int batch_size, double ac_target);
void update_sigma_e(sigma_info &si, data_info &di, double nu, double lambda, double* wts, RNG& gen, bool prior_only);
void update_sigma_u(sigma_info& si, data_info &di, double hyperprior, double* wts, RNG &gen, bool prior_only);
void draw_u(sigma_info &si, data_info& di, double *w, RNG &gen, bool prior_only);
void update_mu_scale(sigma_info& si, data_info& di, double* allfit, double* allfit_proposed, double* mu_train, RNG& gen, bool prior_only);
void update_tau_scale(sigma_info& si, data_info& di, double* allfit, double* allfit_proposed, double* tau_train, RNG& gen, bool prior_only);
double calculate_lp_diff_forscales(double* allfit, double* allfit_proposed, double log_prior_current, double log_prior_proposed, data_info& di, bool prior_only);

#endif /* end of sigma_helpers_h */