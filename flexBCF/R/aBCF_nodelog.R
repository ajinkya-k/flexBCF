aBCF_nodelog <- function(..., log_mu_nodes=TRUE, log_tau_nodes=TRUE) {
  logfile <- tempfile()
  
  sink(logfile)
  fit <- aBCF_mc(log_mu_nodes=log_mu_nodes, log_tau_nodes=log_tau_nodes, verbose=FALSE, ...)
  sink(NULL)
  
  fit$log <- dplyr::tibble(line = readLines(logfile)) |>
    mutate(iter = 1 + str_match(line,'iter ([0-9]+)')[,2] |> as.numeric(),
           chain = ifelse(stringr::str_detect(line,'iter 0$'), row_number(), NA) |> as.factor() |> as.numeric(),
           tree = 1 + stringr::str_match(line,'(mu|tau)tree ([0-9]+)')[,3] |> as.numeric(),
           par = stringr::str_match(line,'(mu|tau)tree ([0-9]+)')[,2],
           node = 1 + stringr::str_match(line,'node ([0-9]+)')[,2] |> as.numeric()) %>%
    tidyr::fill(iter, chain, tree, par, node, .direction='down') |>
    dplyr::filter(!stringr::str_detect(line, 'iter|tree|node')) |>
    dplyr::mutate(name = stringr::str_split_fixed(line, ' ', 2)[,1],
                  value = stringr::str_split_fixed(line, ' ', 2)[,2] |> as.numeric()) |>
    tidyr::pivot_wider(id_cols=c(iter, chain, tree, par, node))
  
  return(fit)
}
