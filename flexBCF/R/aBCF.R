#' Fit a single chain of an aBCF model
#' 
#' @param Y_train Vector of outcome values; will be standardized internally.
#' @param treated Vector of indicators for treated units
#' @param obs_weights Vector of weights, used for calculating heteroskedatic errors
#' @param X_cont_mu Matrix of continuous predictors for use in fitting mu trees; must be scaled to [-1,1]
#' @param X_cat_mu Matrix of categorical predictors for mu trees
#' @param X_cont_tau Matrix of continuous predictors for tau trees; must be scaled to [-1,1]
#' @param X_cat_tau Matrix of categorical predictors for tau trees
#' @param unif_cuts_mu Logical vector of length ncol(X_cont_mu) for wheter or not to use uniform cutpoints for continuous mu predictors
#' @param unif_cuts_tau Logical vector of length ncol(X_cont_tau) for wheter or not to use uniform cutpoints for continuous tau predictors
#' @param cutpoints_list_mu List of cutpoints for continuous mu predictors
#' @param cutpoints_list_tau List of cutpoints for continuous tau predictors
#' @param cat_levels_list_mu List of levels for each categorical mu predictor
#' @param cat_levels_list_tau List of levels for each categorical tau predictor
#' @param sparse Logical; if TRUE uses Dirichlet variable selection prior, otherwise uses BART's uniform random variable selection
#' @param M_mu Number of mu trees
#' @param M_tau Number of tau trees
#' @param sd_mu Prior scale of the leaf node means of mu trees
#' @param alpha_mu Base for tree prior on mu trees
#' @param beta_mu Power for tree prior on mu trees
#' @param sd_tau Prior scale of the leaf node means of tau trees
#' @param alpha_tau Base for tree prior on tau trees
#' @param beta_tau Power for tree prior on tau trees
#' @param min_node_size Minimum sample size in each end node. 
#' @param use_halfnormal_scales Logical; whether or not to use halfnormal scale priors for mu and tau SDs
#' @param sigu_hyperprior Standard deviation of half-normal hyperprior for sigma_u, given in units of sd(y)
#' @param nd Number of posterior samples to return
#' @param burn Number of burn-in iterations
#' @param thin Thinning factor for posterior; burn + nd*thin total iterations are performed.
#' @param save_samples Whether or not to return results. No clue why you'd ever set to FALSE
#' @param batch_size Batch size for calculating adaptive MH acceptance of sigmas
#' @param acceptance_target Targeted acceptance rate for adaptive MH
#' @param nu Degrees of freedom in the chisq prior on \eqn{sigma^2}
#' @param lambda Scale parameter in the chisq prior on \eqn{sigma^2}
#' @param prior_only Logical; whether to actually fit model or just draw from prior
#' @param verbose Whether to print output to console
#' @param print_every Print to console every print_every'th iteration
#' @param perm Permutation to use to sort data
#' @param chain_num Chain identifier; only used if called in multi-chain run
#' 
#' @useDynLib flexBCFa
#' @export
aBCF <- function(Y_train,
                 treated,
                 obs_weights,
                 X_cont_mu = matrix(0, nrow = 1, ncol = 1),
                 X_cat_mu = matrix(0, nrow = 1, ncol = 1),
                 X_cont_tau = matrix(0, nrow = 1, ncol = 1),
                 X_cat_tau = matrix(0, nrow = 1, ncol = 1),
                 unif_cuts_mu = rep(TRUE, times = ncol(X_cont_mu)),
                 unif_cuts_tau = rep(TRUE, times = ncol(X_cont_tau)),
                 cutpoints_list_mu = NULL,
                 cutpoints_list_tau = NULL,
                 cat_levels_list_mu = NULL,
                 cat_levels_list_tau = NULL,
                 sparse = FALSE,  
                 M_mu = 200, M_tau = 50,
                 sd_mu=2, alpha_mu = 0.95, beta_mu = 2,
                 sd_tau=1, alpha_tau = 0.25, beta_tau = 3,
                 min_node_size=5,
                 use_halfnormal_scales=TRUE,
                 sigu_hyperprior = 2/3,
                 nd = 1000, burn = 1000, thin = 1, save_samples = TRUE,
                 batch_size = 100, acceptance_target=0.44,
                 nu=3, lambda=NULL,
                 prior_only=FALSE,
                 verbose = TRUE, print_every = floor((nd*thin + burn))/10,
                 perm = NULL,
                 chain_num=1)
{
  
  #Check if df is ordered
  Nt = sum(treated)
  Nc = length(treated) - Nt
  is_ordered <- identical(treated[1:Nc], rep(0, Nc)) & identical(treated[(Nc+1):length(treated)], rep(1, Nt))
  if (is.null(perm) & !is_ordered) {
    perm <- order(treated, decreasing=FALSE)
  } else {
    perm <- 1:length(treated)
  }
  
  #If provided tau Xs for both T and C, subset to just T
  if (!is.null(X_cont_tau) && nrow(X_cont_tau) == Nt+Nc) {
    X_cont_tau <- X_cont_tau[treated==1,]
  }
  if (!is.null(X_cat_tau) && nrow(X_cat_tau) == Nt+Nc) {
    X_cat_tau <- X_cat_tau[treated==1,]
  }
  
  Y_train <- Y_train[perm]
  treated <- treated[perm]
  obs_weights <- obs_weights[perm]
  if (!is.null(X_cont_mu) && !nrow(X_cont_mu)==1) {
    X_cont_mu <- X_cont_mu[perm,]  
  }
  if (!is.null(X_cat_mu) && !nrow(X_cat_mu)==1) {
    X_cat_mu <- X_cat_mu[perm,]  
  }
  #No need to reorder Xs for tau, since perm necessarily preserves the within-treated order, 
  #and we've ensured Xs only have treated units
  #aka order(treated)[treated==1] == order(treated[treated==1])
  
  # Standardize the Y's
  y_mean <- weighted.mean(Y_train, obs_weights)
  y_sd <- sqrt(Hmisc::wtd.var(Y_train, obs_weights))
  std_Y <- (Y_train - y_mean)/y_sd
  if (is.null(lambda)) {
    lambda <- mean(obs_weights) * stats::qchisq(0.1, df = nu)/nu  
    if (verbose) print(paste('lambda is', lambda))
  }
  
  mu0 <- c(0,0)
  tau <- c(sd_mu/sqrt(M_mu), sd_tau/sqrt(M_tau))
  
  graph_split_mu <- rep(FALSE, times = ncol(X_cat_mu))
  graph_split_tau <- rep(FALSE, times = ncol(X_cat_tau))
  adj_support_list_mu <- NULL
  adj_support_list_tau <- NULL
  
  
  start_time <- Sys.time()
  fit <- .aBCF(Y_train = std_Y,
               treated = treated,
               tX_cont_mu_train = t(X_cont_mu),
               tX_cat_mu_train = t(X_cat_mu),
               tX_cont_tau_train = t(X_cont_tau),
               tX_cat_tau_train = t(X_cat_tau),
               obs_weights = obs_weights,
               unif_cuts_mu = unif_cuts_mu,
               unif_cuts_tau = unif_cuts_tau,
               cutpoints_list_mu = cutpoints_list_mu,
               cutpoints_list_tau = cutpoints_list_tau,
               cat_levels_list_mu = cat_levels_list_mu,
               cat_levels_list_tau = cat_levels_list_tau,
               graph_split_mu = graph_split_mu,
               graph_split_tau = graph_split_tau,
               graph_cut_type_mu = 0, graph_cut_type_tau = 0,
               adj_support_list_mu = adj_support_list_mu,
               adj_support_list_tau = adj_support_list_tau,
               sparse = sparse, a_u = 1, b_u = 1,
               mu0 = mu0, tau = tau, #THIS TAU IS PRIOR VAR, NOT TAU FN,SRY!
               lambda = lambda, nu = nu, sigu_hyperprior = sigu_hyperprior,
               M_mu = M_mu, M_tau = M_tau,
               alpha_mu = alpha_mu, beta_mu = beta_mu,
               alpha_tau = alpha_tau, beta_tau = beta_tau,
               min_node_size = min_node_size,
               use_halfnormal_scales=use_halfnormal_scales,
               nd = nd, burn = burn, thin = thin, save_samples = save_samples,
               batch_size = batch_size, acceptance_target = acceptance_target,
               prior_only=prior_only,
               verbose = verbose, print_every = print_every)
  stop_time <- Sys.time()
  
  results <- list()
  results[["chain_num"]]    <- chain_num
  results[["mu_trees"]]     <- fit$mu
  results[["tau_trees"]]    <- fit$tau
  results[["mu"]]           <- fit$mu_fit[,order(perm)]    * y_sd + y_mean
  results[["tau"]]          <- fit$tau_fit[,order(perm)]   * y_sd
  results[["sigma_u"]]      <- fit$sigma_u                 * y_sd
  results[["sigma_y"]]      <- fit$sigma_e                 * y_sd
  results[["u"]]            <- fit$u_samples[,order(perm)] * y_sd
  results[["varcount_mu"]]  <- fit$varcount_mu
  results[["varcount_tau"]] <- fit$varcount_tau
  results[["mu_scale"]]     <- fit$mu_scale
  results[["tau_scale"]]    <- fit$tau_scale
  results[["acceptance"]]   <- fit$acceptance
  results[["y_mean"]]       <- y_mean
  results[["y_sd"]]         <- y_sd
  results[["cat_levels_list"]] <- list(mu = cat_levels_list_mu, tau = cat_levels_list_tau)
  results[["time"]] <- as.numeric(base::difftime(stop_time, start_time, units='secs'))
  
  names(results[["acceptance"]]) <- c('sigma_e', 'sigma_u','mu_scale', 'tau_scale')
  if (!use_halfnormal_scales) {
    results[["acceptance"]] <- results[["acceptance"]][1:2]
  }
  
  return(results)
}