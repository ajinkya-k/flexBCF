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




void draw_leaf_mu_het(tree &t, suff_stat &ss, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool print)
{
  double P;
  double Theta;
  double post_sd;
  double post_mean;
  int inode = -1;
  int cnt;
  double draw;
  tree::tree_p bn; // we are modifying bn so we need a pointer not a constant pointer
  
  for(suff_stat_it ss_it = ss.begin(); ss_it != ss.end(); ++ss_it){
    bn = t.get_ptr(ss_it->first);
    inode += 1;
    if(bn == 0) Rcpp::stop("[draw_mu]: could not find node that is in suff stat map in the tree");
    else{
      P = 1.0/pow(tree_pi.tau, 2.0); // precision of jump mu
      Theta = tree_pi.mu0/pow(tree_pi.tau, 2.0);
      double s2; //container for variance for ease of reading the code, used in loop below
      double scale2 = pow(di.mu_scale, 2.0);
      cnt = 0;
      for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it) {
        s2 = di.var_i[*it];
        Theta += di.rp[*it]*di.mu_scale/s2;
        P += scale2/s2;
        cnt += 1;
      }

      post_sd = sqrt(1.0/P);
      post_mean = Theta/P;
      draw = gen.normal(post_mean, post_sd);

      tree::tree_p prnt = bn->get_p();
      if (print) {
        Rcpp::Rcout << "node " << inode << std::endl;

        if (prnt != 0) {
          Rcpp::Rcout << "rtypeaa " << prnt->get_rule().is_aa << std::endl; // continous variable
          Rcpp::Rcout << "rtypect " << prnt->get_rule().is_cat << std::endl; // categorical rule
          // Rcpp::Rcout << "rtyperc " << prnt->get_rule().c << std::endl;
          if(prnt->get_rule().is_aa) {
            Rcpp::Rcout << "varno " << prnt->get_rule().v_aa << std::endl;
            Rcpp::Rcout << "cutpt " << prnt->get_rule().c << std::endl;
          }
          if(prnt->get_rule().is_cat) {
            Rcpp::Rcout << "varno " << prnt->get_rule().v_cat << std::endl;
            Rcpp::Rcout << "cutpt " << -99 << std::endl;
          }
          Rcpp::Rcout << "lsize " << prnt->get_rule().l_vals.size() << std::endl;
          Rcpp::Rcout << "rsize " << prnt->get_rule().r_vals.size() << std::endl;
        }
        Rcpp::Rcout << "Theta " << Theta << std::endl;
        Rcpp::Rcout << "P " << P << std::endl;
        Rcpp::Rcout << "N " << cnt << std::endl;
        Rcpp::Rcout << "post_mean " << post_mean << std::endl;
        Rcpp::Rcout << "post_sd " << post_sd << std::endl;
        Rcpp::Rcout << "draw " << draw << std::endl;
      }

      bn->set_mu(draw);
    }
  }
}

void draw_leaf_tau_het(tree &t, suff_stat &ss, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool print)
{
  double P;
  double Theta;
  double post_sd;
  double post_mean;
  int inode = -1;
  int cnt;
  double draw;
  tree::tree_p bn; // we are modifying bn so we need a pointer not a constant pointer
  
  for(suff_stat_it ss_it = ss.begin(); ss_it != ss.end(); ++ss_it){
    bn = t.get_ptr(ss_it->first);
    inode += 1;
    if(bn == 0) Rcpp::stop("[draw_mu]: could not find node that is in suff stat map in the tree");
    else{
      P = 1.0/pow(tree_pi.tau, 2.0); // precision of jump mu
      Theta = tree_pi.mu0/pow(tree_pi.tau, 2.0);
      double s2;
      double scale2 = pow(di.tau_scale, 2.0);
      cnt = 0;
      // remember that ss_it->second contains the index within the treated set and not the whole dataset
      // we must offset by di.n_control!
      for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it) {
        s2 = di.var_i[*it + di.n_control];
        // rp is on the actual scale, not unitless. So convert to unitless as
        //rp/m, then multiply byh m^2/s^2
        Theta += di.rp[*it + di.n_control]*di.tau_scale/s2;
        P += scale2/s2;
        cnt += 1;
      }

      post_sd = sqrt(1.0/P);
      post_mean = Theta/P;
      draw = gen.normal(post_mean, post_sd);

      if (print) {
        Rcpp::Rcout << "node " << inode << std::endl;
        Rcpp::Rcout << "Theta " << Theta << std::endl;
        Rcpp::Rcout << "P " << P << std::endl;
        Rcpp::Rcout << "N " << cnt << std::endl;
        Rcpp::Rcout << "post_mean " << post_mean << std::endl;
        Rcpp::Rcout << "post_sd " << post_sd << std::endl;
        Rcpp::Rcout << "draw " << draw << std::endl;
      }

      bn->set_mu(draw);
      
    }
  }
}
