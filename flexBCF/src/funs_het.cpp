#include "funs.h"

//



double compute_lil_mu_het(suff_stat &ss, int &nid, data_info &di, tree_prior_info &tree_pi)
{
  if(ss.count(nid) != 1) Rcpp::stop("[compute_lil]: did not find node in suff stat map!");
  suff_stat_it ss_it = ss.find(nid);
  
  double P = 1.0/pow(tree_pi.tau, 2.0); // precision of jump mu
  double Theta = tree_pi.mu0/pow(tree_pi.tau, 2.0);
  double s2;
  double scale2 = pow(di.mu_scale, 2.0);
  
  for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it) {
    s2 = di.var_i[*it];
    Theta += di.rp[*it]*scale2/s2;
    P += scale2/s2;
  }
  return(-0.5 * log(P) + 0.5 * pow(Theta,2.0) / P);
}

double compute_lil_tau_het(suff_stat &ss, int &nid, data_info &di, tree_prior_info &tree_pi)
{
  if(ss.count(nid) != 1) Rcpp::stop("[compute_lil]: did not find node in suff stat map!");
  suff_stat_it ss_it = ss.find(nid);
  
  double P = 1.0/pow(tree_pi.tau, 2.0); // precision of jump mu
  double Theta = tree_pi.mu0/pow(tree_pi.tau, 2.0);
  double s2;
  double scale2 = pow(di.tau_scale, 2.0);

  // remember that ss_it->second contains the index within the treated set and not the whole dataset
  // we must offset by di.n_control!
  for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it) {
    s2 = di.var_i[*it + di.n_control];
    Theta += di.rp[*it + di.n_control]*scale2/s2;
    P += scale2/s2;
  }
  return(-0.5 * log(P) + 0.5 * pow(Theta,2.0) / P);
}




void draw_leaf_mu_het(tree &t, suff_stat &ss, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool prior_only)
{
  double P;
  double Theta;
  double post_sd;
  double post_mean;
  tree::tree_p bn; // we are modifying bn so we need a pointer not a constant pointer
  
  for(suff_stat_it ss_it = ss.begin(); ss_it != ss.end(); ++ss_it){
    bn = t.get_ptr(ss_it->first);
    if(bn == 0) Rcpp::stop("[draw_mu]: could not find node that is in suff stat map in the tree");
    else{
      P = 1.0/pow(tree_pi.tau, 2.0); // precision of jump mu
      Theta = tree_pi.mu0/pow(tree_pi.tau, 2.0);
      double s2; //container for variance for ease of reading the code, used in loop below
      double scale2 = pow(di.mu_scale, 2.0);

      if (!prior_only) {
        for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it) {
          s2 = di.var_i[*it];
          Theta += di.rp[*it]*di.mu_scale/s2;
          P += scale2/s2;
        }
      }

      post_sd = sqrt(1.0/P);
      post_mean = Theta/P;
      bn->set_mu(gen.normal(post_mean, post_sd));
    }
  }
}

void draw_leaf_tau_het(tree &t, suff_stat &ss, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool prior_only)
{
  double P;
  double Theta;
  double post_sd;
  double post_mean;
  tree::tree_p bn; // we are modifying bn so we need a pointer not a constant pointer
  
  for(suff_stat_it ss_it = ss.begin(); ss_it != ss.end(); ++ss_it){
    bn = t.get_ptr(ss_it->first);
    if(bn == 0) Rcpp::stop("[draw_mu]: could not find node that is in suff stat map in the tree");
    else{
      P = 1.0/pow(tree_pi.tau, 2.0); // precision of jump mu
      Theta = tree_pi.mu0/pow(tree_pi.tau, 2.0);
      double s2;
      double scale2 = pow(di.tau_scale, 2.0);

      if (!prior_only) {
        // remember that ss_it->second contains the index within the treated set and not the whole dataset
        // we must offset by di.n_control!
        for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it) {
          s2 = di.var_i[*it + di.n_control];
          // rp is on the actual scale, not unitless. So convert to unitless as
          //rp/m, then multiply byh m^2/s^2
          Theta += di.rp[*it + di.n_control]*di.tau_scale/s2;
          P += scale2/s2;
        }
      }

      post_sd = sqrt(1.0/P);
      post_mean = Theta/P;
      bn->set_mu(gen.normal(post_mean, post_sd));
      
    }
  }
}
