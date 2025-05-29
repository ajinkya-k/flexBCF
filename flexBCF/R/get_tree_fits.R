#' Generate predictions from a fitted flexBCFa model
#' 
#' @param fit The fitted model object
#' @param type Character vector of which type of trees (mu, tau) to generate prediction for.
#' @param X_cont Matrix of continuous predictors to use. If you want to use different predictors
#'  for mu vs tau you will need to run predict separately
#'  NB: predictors must be scaled to [-1,1]; this must be done using the same scaling as was used for fitting the original model
#' @param X_cat Matrix of categorical predictors
#' @param verbose Whether to print verbosely
#' @param print_every Print to log after each print_every'th iteration
#' @param n_cores Number of cores to use for prediction. Should be <= number of chains
#' @export
get_tree_fits <- function(fit, 
                          type = c("mu","tau"),
                          X_cont = matrix(0, nrow = 1, ncol = 1),
                          X_cat = matrix(0, nrow = 1, ncol = 1),
                          verbose = TRUE, 
                          print_every = floor(nrow(fit$mu)/10),
                          n_cores = 1)
{
  if (is.null(fit$n_chains)) {
    n_chains <- 1
    #Convert output from a run without chains to match chain-style output for simplicity
    fit$mu_trees  <- list(fit$mu_trees)
    fit$tau_trees <- list(fit$tau_trees)
    fit$mu_scale  <- matrix(fit$mu_scale,  ncol=1)
    fit$tau_scale <- matrix(fit$tau_scale, ncol=1)
  } else {
    n_chains <- fit$n_chains  
  }
  
  if (type=='mu') {
    tree_draws      = fit$mu_trees
    scale           = fit$mu_scale
    treat           = FALSE
    cat_levels_list = fit$cat_levels_list[["mu"]]
  } else if (type=='tau') {
    tree_draws      = fit$tau_trees
    scale           = fit$tau_scale
    treat           = TRUE
    cat_levels_list = fit$cat_levels_list[["tau"]]
  } else {
    stop("type must be one of mu or tau")
  }
  
  n <- max(c(nrow(X_cont), nrow(X_cat)))
  if(n == 1){
    # Things go a bit haywire if we try to make a prediction with only one subject at a time
    # Best to copy the individual row and make predictions twice. 
    # Will support this case later
    stop("Computing average effects with n = 1 subject is not currently supported")
  } 
  
  if (n_cores > 1 & n_chains > 1) {
    future::plan(future::multisession, workers=n_cores)
    if (verbose) {
      print('Running in parallel')  
    }
    mapfun <- furrr::future_map
  } else {
    if (verbose) {
      print('Running in series')  
    }
    mapfun <- purrr::map
  }
  
  tmp <- mapfun(1:n_chains, \(i) {
    .predict_tree_ensemble(tree_draws      = tree_draws[[i]],
                           tX_cont         = t(X_cont),
                           tX_cat          = t(X_cat),
                           scale           = scale[,i],
                           treat           = treat, 
                           y_mean          = fit$y_mean,
                           y_sd            = fit$y_sd,
                           cat_levels_list = cat_levels_list,
                           verbose         = verbose, 
                           print_every     = print_every)
  })
  
  if (n_chains>1) {
    tmp <- abind::abind(tmp, along=3) |> aperm(c(1,3,2))
  } else {
    tmp <- tmp[[1]]
  }
  
  return(tmp)
}