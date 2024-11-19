aBCF_mc <- function(...,
                    seed=1234,
                    n_chains=4, 
                    n_cores=4,
                    verbose=TRUE) {
  
  if (n_cores > 1 & n_chains > 1) {
    future::plan(future::multisession, workers=n_cores)
    if (verbose) {
      print('Running in parallel')  
    }
    fit <- furrr::future_pmap(list(chain_num=1:n_chains), aBCF, verbose=verbose,..., .options=furrr::furrr_options(seed=seed))  
  } else {
    if (verbose) {
      print('Running in series')  
    }
    set.seed(seed)
    fit <- purrr::pmap(list(chain_num=1:n_chains), aBCF, verbose=verbose,...)
  }
  
  results <- list()
  results$sigma_u   <- do.call(what=cbind, lapply(fit, \(x) x$sigma_u))
  results$sigma_e   <- do.call(what=cbind, lapply(fit, \(x) x$sigma_e))
  results$mu_fit    <- do.call(what=abind::abind, list(lapply(fit, \(x) x$mu_fit),  along=3)) |> aperm(c(1,3,2))
  results$tau_fit   <- do.call(what=abind::abind, list(lapply(fit, \(x) x$tau_fit), along=3)) |> aperm(c(1,3,2))
  results$u         <- do.call(what=abind::abind, list(lapply(fit, \(x) x$u),       along=3)) |> aperm(c(1,3,2))
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
  
  return(results)
}
