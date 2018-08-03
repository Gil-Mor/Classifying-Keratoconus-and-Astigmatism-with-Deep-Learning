import export_env_variables
from export_env_variables import *
import defs
from defs import *


# ------------- solvers ----------------
best_iter_max_iter = 50
healthy_kc_cly_best_iter_solver = Solver_Net_parameters(
    solver_type = ADAM_SOLVER_TYPE,
    train_epochs=best_iter_max_iter,
    max_iter=best_iter_max_iter,
    display_iter=5,
    test_interval=5,
    snapshot_iter=best_iter_max_iter,
    lr_stepsize=best_iter_max_iter,
)

recycle_best_iter_max_iter = 50
recycle_healthy_kc_cly_best_iter_solver = Solver_Net_parameters(
    solver_type = ADAM_SOLVER_TYPE,
    train_epochs=recycle_best_iter_max_iter,
    max_iter=recycle_best_iter_max_iter,
    display_iter=5,
    test_interval=5,
    snapshot_iter=recycle_best_iter_max_iter,
    lr_stepsize=recycle_best_iter_max_iter,
)

recycle_from_scratch_max_iter = 400
recycle_healthy_kc_cly_from_scratch_solver = Solver_Net_parameters(
    solver_type = ADAM_SOLVER_TYPE,
    train_epochs=recycle_from_scratch_max_iter,
    max_iter=recycle_from_scratch_max_iter,
    display_iter=20,
    test_interval=20,
    snapshot_iter=recycle_from_scratch_max_iter,
    lr_stepsize=recycle_from_scratch_max_iter,
)

# -------------- modes --------------------
healthy_vs_kc_vs_cly_best_iter_cv = Mode ( # DONE 
    mode=["demo_modes", "healthy_vs_kc_vs_cly_best_iter_cv"],
    targets_and_labels=(("healthy", label_a), ("kc", label_b), ("cly", label_c)),
    save_snapshots=True,
    val_set_fraction=0.05,
    solver_net_parameters = healthy_kc_cly_best_iter_solver,
    dummy=False,
    data_dir=my_model_data,
    plot_title="Healthy vs. KC vs. CLY",
)

recycle_healthy_vs_kc_vs_cly_best_iter_cv = Mode ( # DONE 
    mode=["demo_modes", "recycle_healthy_vs_kc_vs_cly_best_iter_cv"],
    targets_and_labels=(("healthy", label_a), ("kc", label_b), ("cly", label_c)),
    save_snapshots=False,
    val_set_fraction=0.1,
    solver_net_parameters = recycle_healthy_kc_cly_best_iter_solver,
    dummy=False,
    data_dir=my_model_data,
    plot_title="Healthy vs. KC vs. CLY",
)

recycle_healthy_vs_kc_vs_cly_from_scratch = Mode ( # DONE 
    mode=["demo_modes", "recycle_healthy_vs_kc_vs_cly_from_scratch"],
    targets_and_labels=(("healthy", label_a), ("kc", label_b), ("cly", label_c)),
    save_snapshots=False,
    val_set_fraction=0.1,
    solver_net_parameters = recycle_healthy_kc_cly_from_scratch_solver,
    dummy=False,
    data_dir=my_model_data,
    plot_title="Healthy vs. KC vs. CLY Train From Scratch",
)
