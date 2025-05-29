

#ifndef GUARD_funs_het_h
#define GUARD_funs_het_h

#include "tree.h"





// void compute_suff_stat_grow_mu_het(suff_stat &orig_suff_stat, suff_stat &new_suff_stat, int &nx_nid, rule_t &rule, tree &t, data_info &di);
// void compute_suff_stat_grow_tau_het(suff_stat &orig_suff_stat, suff_stat &new_suff_stat, int &nx_nid, rule_t &rule, tree &t, data_info &di);
// void compute_suff_stat_prune_het(suff_stat &orig_suff_stat, suff_stat &new_suff_stat, int &nl_nid, int &nr_nid, int &np_nid, tree &t, data_info &di);

double compute_lil_mu_het(suff_stat &ss, int &nid, data_info &di, tree_prior_info &tree_pi);
double compute_lil_tau_het(suff_stat &ss, int &nid, data_info &di, tree_prior_info &tree_pi);





void draw_leaf_mu_het(tree &t, suff_stat &ss, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool prior_only);
void draw_leaf_tau_het(tree &t, suff_stat &ss, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool prior_only);

#endif /* funs_h */
