# KAN for voice pathology detection - research repository

- **kan_toy_test.py** - run this as a first script after installation of requirements.txt to check that PyKAN + Torch are installed correctly
- **kan_arch_mc.py** - script that performs grid search for multiple architectures across multiple datasets, it is the first step in the search for a suitable model
- **analyze_results.py** - script that computes average UAR for Monte Carlo cross validation results, produced by **kan_arch_mc.py**
- **scp_download.bat** - Windows batch file to download results from Blade/whatever
- **kan_arch_best_candidate_mc_men_1.py** - script that utilize the best model found by **kan_arch_mc.py** script. The model is tweaked and tweaked models are evaluated on the best dataset.
- **kan_arch_best_candidate_mc_men_2.py** - script that utilize the best model found by **kan_arch_best_candidate_mc_men_1.py script. The model is tweaked again and tweaked models are evaluated.
- **experiments_plot.py** - script to plot metrics and losses from pickled results