#ifndef GUARD_update_tree_het_h
#define GUARD_update_tree_het_h

#include "funs.h"
#include "funs_het.h"

void grow_tree_mu_het(tree &t, suff_stat &ss_train, int &accept, data_info &di_train, tree_prior_info &tree_pi, RNG &gen, bool prior_only);
void grow_tree_tau_het(tree &t, suff_stat &ss_train, int &accept, data_info &di_train, tree_prior_info &tree_pi, RNG &gen, bool prior_only);


void prune_tree_mu_het(tree &t, suff_stat &ss_train, int &accept, data_info &di_train, tree_prior_info &tree_pi, RNG &gen, bool prior_only);
void prune_tree_tau_het(tree &t, suff_stat &ss_train, int &accept, data_info &di_train, tree_prior_info &tree_pi, RNG &gen, bool prior_only);

void update_tree_mu_het(tree &t, suff_stat &ss_train, int &accept, data_info &di_train, tree_prior_info &tree_pi, RNG &gen, bool prior_only);
void update_tree_tau_het(tree &t, suff_stat &ss_train, int &accept, data_info &di_train, tree_prior_info &tree_pi, RNG &gen, bool prior_only);



#endif /* update_tree_h */
