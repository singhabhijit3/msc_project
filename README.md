# msc_project

The first baseline model is in the file "baseline_1_for_gpu.py". The second baseline model is in "baseline_2_zaremba_for_gpu.py". The main MoS model is in "mos_model_1.py".

The other .py files are extra copies of the three models. These were made so that I could run multiple different hyper-parameter combinations at the same time.

The .sh files are bash scripts to run the .py files on the University of Edinburgh's MLP GPU cluster.

"key_results.txt" and "mos_results.txt" are some rough notes I made while running the models. They have the results of some of the experiments.

The PTB data is stored in the "data" folder. Some of the results (which were close to the average values obtained) are available in "exp_results" folder. I did not save all results because it would have taken too much storage space.

Please note that the hyper-parameter combinations in the .py files are just some of the last experiments I ran, and NOT the best performing configurations.

The best hyper-parameter configuration for the MoS model was: num_steps = 70 (context length), batch_size = 5, n_experts = 4, hidden_size = 700, dropout = 0.4, Optimizer = SGD with lr = 20 and clipnorm = 0.25.
