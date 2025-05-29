#' Fit a multi-chain aBCF model
#' 
#' @param seed Random seed
#' @param n_chains Number of chains to run
#' @param n_cores Number of cores to run chains in parallel
#' 
#' The remainder are the arguments to aBCF:
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
#' @export
aBCF_mc <- function(...,
                    seed=1234,
                    n_chains=4, 
                    n_cores=4,
                    verbose=TRUE) {
  start_time <- Sys.time()
  
  if (n_cores > 1 & n_chains > 1) {
    future::plan(future::multisession, workers=n_cores)
    if (verbose) {
      print('Running in parallel')  
    }
    fit <- furrr::future_pmap(list(chain_num=1:n_chains), aBCF, verbose=verbose,..., .options=furrr::furrr_options(seed=seed))
    future::plan(future::sequential)
  } else {
    if (verbose) {
      print('Running in series')  
    }
    set.seed(seed)
    fit <- purrr::pmap(list(chain_num=1:n_chains), aBCF, verbose=verbose,...)
  }
  
  results <- list()
  results$sigma_u   <- do.call(what=cbind, lapply(fit, \(x) x$sigma_u))
  results$sigma_y   <- do.call(what=cbind, lapply(fit, \(x) x$sigma_y))
  results$mu        <- do.call(what=abind::abind, list(lapply(fit, \(x) x$mu),  along=3)) |> aperm(c(1,3,2))
  results$tau       <- do.call(what=abind::abind, list(lapply(fit, \(x) x$tau), along=3)) |> aperm(c(1,3,2))
  results$u         <- do.call(what=abind::abind, list(lapply(fit, \(x) x$u),   along=3)) |> aperm(c(1,3,2))
  results$mu_scale  <- do.call(what=cbind, lapply(fit, \(x) x$mu_scale))
  results$tau_scale <- do.call(what=cbind, lapply(fit, \(x) x$tau_scale))
  results$y_sd      <- fit[[1]]$y_sd
  results$y_mean    <- fit[[1]]$y_mean
  results$n_chains  <- n_chains
  results$seed      <- seed
  
  results$mu_trees         <- lapply(fit, \(x) x$mu_trees)
  results$tau_trees        <- lapply(fit, \(x) x$tau_trees)
  results$varcount_mu      <- lapply(fit, \(x) x$varcount_mu)
  results$varcount_tau     <- lapply(fit, \(x) x$varcount_tau)
  results$cat_levels_list  <- fit[[1]]$cat_levels_list
  results$acceptance       <- do.call(what=rbind, lapply(fit, \(x) x$acceptance))
  colnames(results$acceptance) <- names(fit[[1]]$acceptance)
  results$time <- lapply(fit, \(x) x$time) |> unlist()
  names(results$time) <- paste0('chain', 1:n_chains)
  stop_time <- Sys.time()
  results$time[['overall']] <- as.numeric(base::difftime(stop_time, start_time, units='secs'))
  
  return(results)
}
