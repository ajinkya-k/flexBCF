

#include "update_tree_het.h"
#include "sigma_helpers.h"

// convention: all control subjects listed first

// [[Rcpp::export(".aBCF")]]
Rcpp::List aBCF(Rcpp::NumericVector Y_train,
                   Rcpp::IntegerVector treated,
                   Rcpp::NumericMatrix tX_cont_mu_train,
                   Rcpp::IntegerMatrix tX_cat_mu_train,
                   Rcpp::NumericMatrix tX_cont_tau_train,
                   Rcpp::IntegerMatrix tX_cat_tau_train,
                   Rcpp::NumericVector obs_weights,
                   Rcpp::LogicalVector unif_cuts_mu,
                   Rcpp::LogicalVector unif_cuts_tau,
                   Rcpp::Nullable<Rcpp::List> cutpoints_list_mu,
                   Rcpp::Nullable<Rcpp::List> cutpoints_list_tau,
                   Rcpp::Nullable<Rcpp::List> cat_levels_list_mu,
                   Rcpp::Nullable<Rcpp::List> cat_levels_list_tau,
                   Rcpp::LogicalVector graph_split_mu,
                   Rcpp::LogicalVector graph_split_tau,
                   int graph_cut_type_mu,
                   int graph_cut_type_tau,
                   Rcpp::Nullable<Rcpp::List> adj_support_list_mu,
                   Rcpp::Nullable<Rcpp::List> adj_support_list_tau,
                   bool sparse, double a_u, double b_u,
                   Rcpp::NumericVector mu0, Rcpp::NumericVector tau,
                   double lambda, double nu, double sigu_hyperprior,
                   int M_mu, int M_tau,
                   double alpha_mu, double beta_mu,
                   double alpha_tau, double beta_tau,
                   int nd, int burn, int thin, bool save_samples,
                   int batch_size, double acceptance_target,
                   bool verbose, int print_every)
{
  Rcpp::RNGScope scope;
  RNG gen;
  
  set_str_conversion set_str; // for converting sets of integers into strings
  
  int n_train = 0;
  int n_treat = 0;
  int p_cont_mu = 0;
  int p_cont_tau = 0;
  int p_cat_mu = 0;
  int p_cat_tau = 0;
  
  parse_training_data(n_train, p_cont_mu, p_cat_mu, tX_cont_mu_train, tX_cat_mu_train);
  parse_training_data(n_treat, p_cont_tau, p_cat_tau, tX_cont_tau_train, tX_cat_tau_train);
  int n_control = n_train - n_treat;
  
  if(Y_train.size() != n_train) Rcpp::stop("Number of observations in Y_train does not match number of rows in training design matrices");
  
  int p_mu = p_cont_mu + p_cat_mu;
  int p_tau = p_cont_tau + p_cat_tau;
  
  if(verbose){
    Rcpp::Rcout << "n_train = " << n_train << " n_treat = " << n_treat << " n_control = " << n_control;
    Rcpp::Rcout << " p_cont_mu = " << p_cont_mu << "  p_cat_mu = " << p_cat_mu << std::endl;
    Rcpp::Rcout << " p_cont_tau = " << p_cont_tau << "  p_cat_tau = " << p_cat_tau << std::endl;
  }
  
  std::vector<std::set<double>> cutpoints_mu;
  std::vector<std::set<int>> cat_levels_mu;
  std::vector<int> K_mu; // number of levels for the different categorical variables
  std::vector<std::vector<unsigned int>> adj_support_mu;
  
  if(p_cont_mu > 0){
    if(cutpoints_list_mu.isNotNull()){
      Rcpp::List tmp_cutpoints = Rcpp::List(cutpoints_list_mu);
      parse_cutpoints(cutpoints_mu, p_cont_mu, tmp_cutpoints, unif_cuts_mu);
    }
  }
  
  if(p_cat_mu > 0){
    if(cat_levels_list_mu.isNotNull()){
      Rcpp::List tmp_cat_levels = Rcpp::List(cat_levels_list_mu);
      parse_cat_levels(cat_levels_mu, K_mu, p_cat_mu, tmp_cat_levels);
    } else{
      Rcpp::stop("Must provide categorical levels.");
    }
    if(adj_support_list_mu.isNotNull()){
      Rcpp::List tmp_adj_support = Rcpp::List(adj_support_list_mu);
      parse_cat_adj(adj_support_mu, p_cat_mu, tmp_adj_support, graph_split_mu);
    }
  }
  
  std::vector<std::set<double>> cutpoints_tau;
  std::vector<std::set<int>> cat_levels_tau;
  std::vector<int> K_tau; // number of levels for the different categorical variables
  std::vector<std::vector<unsigned int>> adj_support_tau;
  
  if(p_cont_tau > 0){
    if(cutpoints_list_tau.isNotNull()){
      Rcpp::List tmp_cutpoints = Rcpp::List(cutpoints_list_tau);
      parse_cutpoints(cutpoints_tau, p_cont_tau, tmp_cutpoints, unif_cuts_tau);
    }
  }
  
  if(p_cat_tau > 0){
    if(cat_levels_list_tau.isNotNull()){
      Rcpp::List tmp_cat_levels = Rcpp::List(cat_levels_list_tau);
      parse_cat_levels(cat_levels_tau, K_tau, p_cat_tau, tmp_cat_levels);
    } else{
      Rcpp::stop("Must provide categorical levels.");
    }
    if(adj_support_list_tau.isNotNull()){
      Rcpp::List tmp_adj_support = Rcpp::List(adj_support_list_tau);
      parse_cat_adj(adj_support_tau, p_cat_tau, tmp_adj_support, graph_split_tau);
    }
  }
  
  
  double* allfit_train = new double[n_train];
  double* var_i = new double[n_train];
  double* mu_train = new double[n_train]; // temp container for mu_i
  double* tau_train = new double[n_train]; // temp container for tau_i only for treated! 
  double* u_vec = new double[n_train];
  double* residual = new double[n_train];
  
  
  // set up our data info object
  data_info di_train;
  di_train.n = n_train;
  di_train.n_control = n_control;
  di_train.n_treat = n_treat;
  di_train.p_cont_mu = p_cont_mu;
  di_train.p_cat_mu = p_cat_mu;
  di_train.p_mu = p_mu;
  di_train.p_cont_tau = p_cont_tau;
  di_train.p_cat_tau = p_cat_tau;
  di_train.p_tau = p_tau;
  di_train.treated = treated.begin();
  di_train.var_i = var_i;
  
  if(p_cont_mu > 0) di_train.x_cont_mu = tX_cont_mu_train.begin();
  if(p_cont_tau > 0) di_train.x_cont_tau = tX_cont_tau_train.begin();
  if(p_cat_mu > 0) di_train.x_cat_mu = tX_cat_mu_train.begin();
  if(p_cat_tau > 0) di_train.x_cat_tau = tX_cat_tau_train.begin();
  di_train.rp = residual;

  Rcpp::Rcout << "Setting up sigma info" << std::endl;
  sigma_info s_info;

  double y2_sum = 0.0;
  double ybar = 0.0;

  for (int i = 0; i < n_train; i++) {
    ybar += Y_train[i];
    y2_sum += Y_train[i] * Y_train[i];
  }

  ybar /= n_train;
  s_info.sigma_e = sqrt((y2_sum-n_train*ybar*ybar)/(n_train-1));
  s_info.sigma_u = fabs(gen.normal(0., sigu_hyperprior));
  s_info.ls_sigma_e = 0.0;
  s_info.ls_sigma_u = 0.0;
  s_info.ac_sigma_e = 0;
  s_info.ac_sigma_u = 0;

  double* prop_var_i = new double[n_train];

  s_info.prop_var_i = prop_var_i;
  s_info.u_sample = u_vec;

  double* wts = new double[n_train];
  for (int i = 0; i < n_train; i++) wts[i] = obs_weights[i];

  Rcpp::Rcout << "Calculating sigma_i for the first time" << std::endl;
  //TODO: init sigmas
  calculate_sigma2_i(s_info.sigma_e, s_info.sigma_u, di_train.n, wts, var_i);
  // stuff for variable selection
  std::vector<double> theta_mu(p_mu, 1.0/ (double) p_mu);
  std::vector<double> theta_tau(p_tau, 1.0/ (double) p_tau);
  
  double u_mu = 1.0/(1.0 + (double) p_mu);
  double u_tau = 1.0/(1.0 + (double) p_tau);
  
  std::vector<int> var_count_mu(p_mu, 0); // count how many times a variable has been used in a splitting rule
  std::vector<int> var_count_tau(p_tau, 0);
  
  int rule_count_mu = 0; // how many total decision rules are there in the ensemble
  int rule_count_tau = 0;
  
  int rc_rule_count_mu = 0;
  int rc_rule_count_tau = 0;

  int rc_var_count_mu = 0;
  int rc_var_count_tau = 0;
  
  tree_prior_info tree_pi_mu;
  tree_pi_mu.theta = &theta_mu;
  tree_pi_mu.alpha = alpha_mu;
  tree_pi_mu.beta = beta_mu;
  tree_pi_mu.var_count = &var_count_mu;
  tree_pi_mu.rule_count = &rule_count_mu;
  tree_pi_mu.rc_rule_count = &rc_rule_count_mu;
  tree_pi_mu.rc_var_count = &rc_var_count_mu;

  tree_pi_mu.unif_cuts = unif_cuts_mu.begin(); // do we use uniform cutpoints?
  tree_pi_mu.cutpoints = &cutpoints_mu;

  tree_pi_mu.cat_levels = &cat_levels_mu;
  tree_pi_mu.K = &K_mu;
  tree_pi_mu.adj_support = &adj_support_mu;
  
  tree_pi_mu.graph_split = graph_split_mu.begin();
  tree_pi_mu.graph_cut_type = graph_cut_type_mu;
  tree_pi_mu.mu0 = mu0[0];
  tree_pi_mu.tau = tau[0];
  
  tree_prior_info tree_pi_tau;
  tree_pi_tau.alpha = alpha_tau;
  tree_pi_tau.beta = beta_tau;
  tree_pi_tau.theta = &theta_tau;
  tree_pi_tau.var_count = &var_count_tau;
  tree_pi_tau.rule_count = &rule_count_tau;
  tree_pi_tau.rc_rule_count = &rc_rule_count_tau;
  tree_pi_tau.rc_var_count = &rc_var_count_tau;
  
  tree_pi_tau.unif_cuts = unif_cuts_tau.begin(); // do we use uniform cutpoints?
  tree_pi_tau.cutpoints = &cutpoints_tau;
  
  tree_pi_tau.cat_levels = &cat_levels_tau;
  tree_pi_tau.K = &K_tau;
  tree_pi_tau.adj_support = &adj_support_tau;
  
  tree_pi_tau.graph_split = graph_split_tau.begin();
  tree_pi_tau.graph_cut_type = graph_cut_type_tau;
  tree_pi_tau.mu0 = mu0[1];
  tree_pi_tau.tau = tau[1];
  
  
  if (verbose) {
    Rcpp::Rcout << "For  mu trees: alpha = " << tree_pi_mu.alpha << ", beta = " << tree_pi_mu.beta << std::endl;
    Rcpp::Rcout << "For tau trees: alpha = " << tree_pi_tau.alpha << ", beta = " << tree_pi_tau.beta << std::endl;
  }
  // stuff for sigma
  // double sigma_e = 1.0; // parameter for error variance
  // double sigma_u = 1.0; // parameter for RE variance
  // double* total_sq_resid_i = new double[n_train]; //instead of double total_sq_resid = 0.0,  sum of squared residuals
  for(int i = 0; i < n_train; i++) var_i[i] = 1.0;
  // for(int i = 0; i < n_train; i++) total_sq_resid_i[i] = 0.0;
  // double scale_post = 0.0;
  // double nu_post = 0.0;
  

  // stuff for MCMC loop
  int total_draws = 1 + burn + (nd-1)*thin;
  int sample_index = 0;
  int accept = 0;
  int total_accept = 0; // counts how many trees we change in each iteration
  tree::npv bnv; // for checking that our ss map and our trees are not totally and utterly out of sync
  double tmp_mu; // for holding the value of mu when we're doing the backfitting
  
  // initialize the trees
  std::vector<tree> t_mu_vec(M_mu);
  std::vector<tree> t_tau_vec(M_tau);

  std::vector<suff_stat> ss_train_mu_vec(M_mu);
  std::vector<suff_stat> ss_train_tau_vec(M_tau);
    
  Rcpp::Rcout << "Initial traversal" << std::endl;
  for (int i = 0; i < n_train; i++) allfit_train[i] = 0.0;
  
  // initial tree traversal for mu
  for(int m = 0; m < M_mu; m++){
    tree_traversal_mu(ss_train_mu_vec[m], t_mu_vec[m], di_train);
    for(suff_stat_it ss_it = ss_train_mu_vec[m].begin(); ss_it != ss_train_mu_vec[m].end(); ++ss_it){
      tmp_mu = t_mu_vec[m].get_ptr(ss_it->first)->get_mu(); // get the value of mu in the leaf
      for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it) allfit_train[*it] += tmp_mu;
    }
  }
  
  for(int m = 0; m < M_tau; m++){
    tree_traversal_tau(ss_train_tau_vec[m], t_tau_vec[m], di_train);
    for(suff_stat_it ss_it = ss_train_tau_vec[m].begin(); ss_it != ss_train_tau_vec[m].end(); ++ss_it){
      tmp_mu = t_tau_vec[m].get_ptr(ss_it->first)->get_mu(); // get the value of mu in the leaf
      for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it) allfit_train[*it + n_control] += tmp_mu;
    }
  }

  for(int i = 0; i < n_train; i++) residual[i] = Y_train[i] - allfit_train[i];
  
  // output containers
  Rcpp::List mu_tree_draws(nd);
  Rcpp::List tau_tree_draws(nd);
  // Rcpp::NumericVector sigma_samples(total_draws);
  arma::mat var_count_samples_mu(total_draws,p_mu);
  arma::mat var_count_samples_tau(total_draws, p_tau);
  arma::mat mu_fit_samples = arma::zeros<arma::mat>(1, 1);
  arma::mat tau_fit_samples = arma::zeros<arma::mat>(1, 1);
  arma::mat u_samples = arma::zeros<arma::mat>(1, 1);
  Rcpp::NumericVector sigma_u_samples(nd);
  Rcpp::NumericVector sigma_e_samples(nd);
  
  if (save_samples) {
    mu_fit_samples.zeros(nd,n_train);
    tau_fit_samples.zeros(nd,n_train);
    u_samples.zeros(nd, n_train);
  }
  
  
  Rcpp::Rcout << "Starting MCMC" << std::endl;
  // main MCMC loop goes here
  for(int iter = 0; iter < total_draws; iter++){
    if(verbose){
      if( (iter < burn) && (iter % print_every == 0)){
        Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << total_draws << "; Burn-in" << std::endl;
        Rcpp::checkUserInterrupt();
      } else if(((iter> burn) && (iter%print_every == 0)) || (iter == burn)){
        Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << total_draws << "; Sampling" << std::endl;
        Rcpp::checkUserInterrupt();
      }
    }
    
    // loop over the mu trees first
    total_accept = 0;
    for (int i = 0; i < n_train; i++) {
      mu_train[i] = 0.0;
      tau_train[i] = 0.0;
    }
    for(int m = 0; m < M_mu; m++){
      for(suff_stat_it ss_it = ss_train_mu_vec[m].begin(); ss_it != ss_train_mu_vec[m].end(); ++ss_it){
        // loop over the bottom nodes in m-th tree
        tmp_mu = t_mu_vec[m].get_ptr(ss_it->first)->get_mu(); // get the value of mu in the leaf
        for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it){
          // remove fit of m-th tree from allfit: allfit[i] -= tmp_mu
          // for partial residual: we could compute Y - allfit (now that allfit has fit of m-th tree removed)
          // numerically this is exactly equal to adding tmp_mu to the value of residual
          allfit_train[*it] -= tmp_mu; // adjust the value of allfit
          residual[*it] += tmp_mu;
        }
      } // this whole loop is O(n)
      
      update_tree_mu_het(t_mu_vec[m], ss_train_mu_vec[m], accept, di_train, tree_pi_mu, gen); // update the tree
      total_accept += accept;
  
      // now we need to update the value of allfit
      for(suff_stat_it ss_it = ss_train_mu_vec[m].begin(); ss_it != ss_train_mu_vec[m].end(); ++ss_it){
        tmp_mu = t_mu_vec[m].get_ptr(ss_it->first)->get_mu();
        for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it){
          // add fit of m-th tree back to allfit and subtract it from the value of the residual
          allfit_train[*it] += tmp_mu;
          residual[*it] -= tmp_mu;
          mu_train[*it] += tmp_mu;
        }
      } // this loop is also O(n)
    } // closes loop over all of the trees
    
    
    // now it's time to update the tau_trees
    for(int m = 0; m < M_tau; m++){
      for(suff_stat_it ss_it = ss_train_tau_vec[m].begin(); ss_it != ss_train_tau_vec[m].end(); ++ss_it){
        // loop over the bottom nodes in m-th tree
        tmp_mu = t_tau_vec[m].get_ptr(ss_it->first)->get_mu(); // get the value of mu in the leaf
        for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it){
          // remove fit of m-th tree from allfit: allfit[i] -= tmp_mu
          // for partial residual: we could compute Y - allfit (now that allfit has fit of m-th tree removed)
          // numerically this is exactly equal to adding tmp_mu to the value of residual
          // sinc this is a loop only over the treated subjects, we need to offset things in allfit and residual
          allfit_train[*it + n_control] -= tmp_mu; // adjust the value of allfit
          residual[*it + n_control] += tmp_mu;
        }
      } // this whole loop is O(n)
      
      update_tree_tau_het(t_tau_vec[m], ss_train_tau_vec[m], accept, di_train, tree_pi_tau, gen); // update the tree
      total_accept += accept;
  
      // now we need to update the value of allfit
      for(suff_stat_it ss_it = ss_train_tau_vec[m].begin(); ss_it != ss_train_tau_vec[m].end(); ++ss_it){
        tmp_mu = t_tau_vec[m].get_ptr(ss_it->first)->get_mu();
        for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it){
          // add fit of m-th tree back to allfit and subtract it from the value of the residual
          allfit_train[*it + n_control] += tmp_mu;
          residual[*it + n_control] -= tmp_mu;
          tau_train[*it + n_control] += tmp_mu; // only generates tau for treated!
        }
      } // this loop is also O(n)
    } // closes loop over all of the trees
    
    
    // ready to update sigma
    update_sigma_e(s_info, di_train, nu, lambda, wts, gen);
    update_sigma_u(s_info, di_train, sigu_hyperprior, wts, gen);
    draw_u(s_info, di_train, wts, gen);

    if ((iter+1) % batch_size == 0) {
      update_adaptive_ls(s_info, iter, batch_size, acceptance_target);
    }
    if(sparse){
      update_theta_u(theta_mu, u_mu, var_count_mu, p_mu, a_u, b_u, gen);
      update_theta_u(theta_tau, u_tau, var_count_tau, p_tau, a_u, b_u, gen);
    }
    for(int j = 0; j < p_mu; j++) var_count_samples_mu(iter,j) = var_count_mu[j];
    for(int j = 0; j < p_tau; j++) var_count_samples_tau(iter,j) = var_count_tau[j];
    
    if( (iter >= burn) && ( (iter - burn)%thin == 0)){

      // Rcpp::Rcout << "Saving samples" << std::endl;
      sample_index = (int) ( (iter-burn)/thin);
      
      Rcpp::CharacterVector mu_tree_string_vec(M_mu);
      Rcpp::CharacterVector tau_tree_string_vec(M_tau);
      
      for(int m = 0; m < M_mu; m++) mu_tree_string_vec[m] = write_tree(t_mu_vec[m], tree_pi_mu, set_str);
      for(int m = 0; m < M_tau; m++) tau_tree_string_vec[m] = write_tree(t_tau_vec[m], tree_pi_tau, set_str);
      mu_tree_draws[sample_index] = mu_tree_string_vec;
      tau_tree_draws[sample_index] = tau_tree_string_vec; // dump a character vector holding each tree's draws into an element of an Rcpp::List
      
      for (int i = 0; i < n_train; i++) {
        mu_fit_samples(sample_index, i) = mu_train[i];
        // Rcpp::Rcout << i <<  ": Saved mu fit,  ";
        tau_fit_samples(sample_index, i) = tau_train[i];
        // Rcpp::Rcout << "Saved tau fit, " ;
        u_samples(sample_index, i) = u_vec[i];
        // Rcpp::Rcout << "Saved u draw. " ;
      }

      sigma_u_samples(sample_index) = s_info.sigma_u;
      // Rcpp::Rcout << "Saved sig_u fit" << std::endl;
      sigma_e_samples(sample_index) = s_info.sigma_e;

      // Rcpp::Rcout << "Saved sig_e fit" << std::endl;
      /*
      //  for debugging purposes only!
      for(int m = 0; m < M_tau; m++){
        for(suff_stat_it ss_it = ss_train_tau_vec[m].begin(); ss_it != ss_train_tau_vec[m].end(); ++ss_it){
          tmp_mu = t_tau_vec[m].get_ptr(ss_it->first)->get_mu();
          for(int_it it = ss_it->second.begin(); it != ss_it->second.end(); ++it){
            tau_fit_samples(sample_index,*it) += tmp_mu;
          }
        }
      }
      
       */
    } // closes if that checks whether we should save anything in this iteration
  } // closes the main MCMC for loop

  Rcpp::List results;
  results["sigma_u"] = sigma_u_samples;
  results["sigma_e"] = sigma_e_samples;
  results["u_samples"] = u_samples;
  results["mu"] = mu_tree_draws;
  results["tau"] = tau_tree_draws;
  results["mu_fit"] = mu_fit_samples;
  results["tau_fit"] = tau_fit_samples;
  results["varcount_mu"] = var_count_samples_mu;
  results["varcount_tau"] = var_count_samples_tau;
  //results["tau_fit"] = tau_fit_samples;
  return results;
}
