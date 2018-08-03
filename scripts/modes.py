  
    modes_dir = "ec2_modes" if PLATFORM==PC_Platform else ""

    solver_for_logs_on_gpu = Solver_Net_parameters(
            train_epochs=50,
            max_iter=50,
            display_iter=5,
            test_interval=5,
            snapshot_iter=5,
            lr_stepsize=50,
    )

    solver_for_big_data_on_gpu = Solver_Net_parameters(
        train_epochs=100,
        max_iter=100,
        display_iter=10,
        test_interval=10,
        snapshot_iter=10,
        lr_stepsize=200,
        )

    adam_solver_for_best_iter_for_hkc = Solver_Net_parameters(
            solver_type=ADAM_SOLVER_TYPE,
            train_epochs=25,
            max_iter=25,
            display_iter=5,
            test_interval=5,
            snapshot_iter=5,
            lr_stepsize=25,
    )
    test_every_epoch = Solver_Net_parameters(
            train_epochs=15,
            max_iter=15,
            display_iter=1,
            test_interval=1,
            snapshot_iter=1,
            lr_stepsize=50,
    )

    test_every_second_epoch= Solver_Net_parameters(
        train_epochs=20,
        max_iter=20,
        display_iter=2,
        test_interval=2,
        snapshot_iter=2,
        lr_stepsize=50,
    )
    adam_solver_for_logs_on_gpu = copy.deepcopy(solver_for_logs_on_gpu)
    adam_solver_for_logs_on_gpu.solver_type = ADAM_SOLVER_TYPE

    adam_solver_test_every_second_epoch = copy.deepcopy(test_every_second_epoch)
    adam_solver_test_every_second_epoch.solver_type = ADAM_SOLVER_TYPE

    adam_solver_for_big_data = copy.deepcopy(solver_for_big_data_on_gpu)
    adam_solver_for_big_data.solver_type = ADAM_SOLVER_TYPE

    
    healthy_vs_kc_cross_validation_mode = Mode( # DONE 
        mode=[modes_dir, "healthy_vs_kc_cross_validation"],
        targets_and_labels=(("healthy", label_a), ("kc", label_b)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        solver_net_parameters=solver_for_logs_on_gpu,
        val_set_fraction=0.05,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC Cross Validation",
    )
    cross_validation_modes.add(healthy_vs_kc_cross_validation_mode)

    
    healthy_vs_kc_cross_validation_adam_one_set_for_prediction = Mode ( # DONE 
        mode=[modes_dir, "healthy_vs_kc_cross_validation_adam_one_set_for_prediction"],
        targets_and_labels=(("healthy", label_a), ("kc", label_b)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        save_snapshots=True,
        val_set_fraction=0.02,
        solver_net_parameters=adam_solver_for_logs_on_gpu,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC Cross Validation - Adam Solver",
    )
    cross_validation_modes.add(healthy_vs_kc_cross_validation_adam_one_set_for_prediction)

    
    healthy_vs_kc_cross_validation_adam = Mode ( # DONE
        mode=[modes_dir, "healthy_vs_kc_cross_validation_adam"],
        targets_and_labels=(("healthy", label_a), ("kc", label_b)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        save_snapshots=False,
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_logs_on_gpu,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC Cross Validation - Adam Solver",
    )
    cross_validation_modes.add(healthy_vs_kc_cross_validation_adam)

    

    healthy_vs_kc_increasing_train_set_size_adam = Mode ( # DONE
        mode=[modes_dir, "healthy_vs_kc_increasing_train_set_size_adam"],
        targets_and_labels=(("healthy", label_a), ("kc", label_b)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        save_snapshots=False,
        val_set_fraction=[0.98, 0.95, 0.9, 0.8, 0.7, 0.5, 0.35, 0.25],
        solver_net_parameters=adam_solver_for_logs_on_gpu,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC Cross Validation - Adam Solver",
    )

    healthy_vs_sus_cross_validation_adam = Mode ( # DONE
        mode=[modes_dir, "healthy_vs_sus_cross_validation_adam"],
        targets_and_labels=(("healthy", label_a), ("sus", label_b)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        save_snapshots=False,
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_big_data,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC Suspects Cross Validation - Adam Solver",
    )
    cross_validation_modes.add(healthy_vs_sus_cross_validation_adam)
    
    healthy_vs_sus_cross_validation_augment_suspects_adam = Mode ( # DONE
        mode=[modes_dir, "healthy_vs_sus_cross_validation_augment_suspects_adam"],
        targets_and_labels=(("healthy", label_a), ("sus", label_b)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        save_snapshots=False,
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_big_data,
        dummy=True,
        augmented_data=True,
        augment_targets=["sus"],
        data_dir=augment_every_image_twice_path,
        plot_title="Healthy vs. KC Suspects Cross Validation - Augmented Suspects - Adam Solver",
    )
    cross_validation_modes.add(healthy_vs_sus_cross_validation_augment_suspects_adam)

    hks_2_classes_cross_validation_20_epochs_test_every_second_epoch_adam = Mode( # DONE
        mode=[modes_dir, "hks_2_classes_cross_validation_20_epochs_test_every_second_epoch_adam"],
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("sus", label_b)),
        save_snapshots=False,
        # txts_data=Txts_data(train_healthy=1, train_kc=1, train_sus=1, val_healthy=1, val_kc=1, val_sus=1, s_label=kc_label),
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_test_every_second_epoch,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC and KC suspects Cross Validation",
    )
    cross_validation_modes.add(hks_2_classes_cross_validation_20_epochs_test_every_second_epoch_adam)
    

    hks_2_classes_cross_validation = Mode( # DONE 
        mode=[modes_dir, "hks_2_classes_cross_validation"],
        save_snapshots=False,
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("sus", label_b)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1, train_sus=1, val_healthy=1, val_kc=1, val_sus=1, s_label=kc_label),
        val_set_fraction=0.05,
        solver_net_parameters=solver_for_logs_on_gpu,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC and KC suspects Cross Validation",
    )
    cross_validation_modes.add(hks_2_classes_cross_validation)
    

    healthy_vs_kc_and_kc_suspects_cross_validation_balanced_classes_adam = Mode( # DONE
        mode=[modes_dir, "hks_2_classes_balanced_cross_validation_adam"],
        save_snapshots=False,
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("sus", label_b)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1, train_sus=1, val_healthy=1, val_kc=1, val_sus=1, s_label=kc_label),
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_logs_on_gpu,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC and KC suspects Balanced Sets Cross Validation Adam Solver",
    )
    cross_validation_modes.add(healthy_vs_kc_and_kc_suspects_cross_validation_balanced_classes_adam)
    

    hks_2_classes_cross_validation_adam = Mode( # DONE
        mode=[modes_dir, "hks_2_classes_cross_validation_adam_solver"],
        save_snapshots=False,
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("sus", label_b)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1, train_sus=1, val_healthy=1, val_kc=1, val_sus=1, s_label=kc_label),
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_logs_on_gpu,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC and KC suspects Cross Validation - Adam Solver",
    )
    cross_validation_modes.add(hks_2_classes_cross_validation_adam)
    

    hks_2_classes_cross_validation_balanced_adam = Mode( # DONE
        mode=[modes_dir, "hks_2_classes_cross_validation_balanced_sets_adam"],
        save_snapshots=False,
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("sus", label_b)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1, train_sus=1, val_healthy=1, val_kc=1, val_sus=1, s_label=kc_label),
        balance_target_names={"keep":"healthy", "throw":"sus"},
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_logs_on_gpu,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC and KC suspects Cross Validation - Balanced Sets Adam Solver"
    )
    cross_validation_modes.add(hks_2_classes_cross_validation_balanced_adam)
    

    healthy_vs_kc_vs_kc_suspects_cross_validation_mode = Mode ( # DONE 
        mode=[modes_dir, "healthy_vs_kc_vs_kc_suspects_cross_validation"],
        save_snapshots=False,
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("sus", label_c)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1, train_sus=1, val_healthy=1, val_kc=1, val_sus=1, s_label=label_c),
        val_set_fraction=0.05,
        solver_net_parameters=solver_for_logs_on_gpu,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC vs. KC suspects Cross Validation",
    )
    cross_validation_modes.add(healthy_vs_kc_vs_kc_suspects_cross_validation_mode)
    

    healthy_vs_kc_vs_kc_suspects_cross_validation_adam = Mode ( # DONE
        mode=[modes_dir, "healthy_vs_kc_vs_kc_suspects_cross_validation_adam"],
        save_snapshots=False,
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("sus", label_c)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1, train_sus=1, val_healthy=1, val_kc=1, val_sus=1, s_label=label_c),
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_logs_on_gpu,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC vs. KC suspects Cross Validation - Adam Solver",
    )
    cross_validation_modes.add(healthy_vs_kc_vs_kc_suspects_cross_validation_adam)
    

    hksc_4_classes_cross_validation_adam = Mode( # DONE
        mode=[modes_dir, "hksc_4_classes_cross_validation_adam"],
        save_snapshots=False,
        targets_and_labels=(("healthy", label_a),("kc", label_b),("sus", label_c),("cly", label_d)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1, train_sus=1, train_cly=1, val_healthy=1, val_kc=1, val_sus=1, val_cly=1,
        #                     s_label=label_c, c_label=label_d),
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_big_data,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC vs. KC suspects vs. CLY Cross Validation - Adam Solver",
    )
    cross_validation_modes.add(hksc_4_classes_cross_validation_adam)

    hksc_4_classes_cross_validation_augmented_sus_and_cly_adam = Mode( # DONE
        mode=[modes_dir, "hksc_4_classes_cross_validation_augmented_sus_and_cly_adam"],
        save_snapshots=False,
        targets_and_labels=(("healthy", label_a),("kc", label_b),("sus", label_c),("cly", label_d)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1, train_sus=1, train_cly=1, val_healthy=1, val_kc=1, val_sus=1, val_cly=1,
        #                     s_label=label_c, c_label=label_d),
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_big_data,
        dummy=True,
        augmented_data=True,
        augment_targets=["sus", "cly"],
        data_dir=augment_every_image_twice_path,
        plot_title="Healthy vs. KC vs. KC suspects vs. CLY Cross Validation with augmented suspects and CLY - Adam Solver",
    )
    cross_validation_modes.add(hksc_4_classes_cross_validation_augmented_sus_and_cly_adam)

    hksc_4_classes_cross_validation_balanced_augmented_data_adam = Mode(  # DONE
        mode=[modes_dir, "hksc_4_classes_cross_validation_augmented_data_adam"],
        save_snapshots=False,
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("sus", label_c), ("cly", label_d)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1, train_sus=1, train_cly=1, val_healthy=1, val_kc=1, val_sus=1, val_cly=1,
        #                     s_label=label_c, c_label=label_d),
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_big_data,
        dummy=True,
        data_dir=augmented_data_path,
        augmented_data=True,
        plot_title="Healthy vs. KC vs. KC suspects vs. CLY Cross Validation with augmented_data- Adam Solver",
    )
    cross_validation_modes.add(hksc_4_classes_cross_validation_adam)

    healthy_vs_kc_vs_cly_cross_validation_adam = Mode ( # DONE 
        mode=[modes_dir, "healthy_vs_kc_vs_cly_cross_validation_adam"],
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("cly", label_c)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        save_snapshots=False,
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_logs_on_gpu,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC vs. CLY Cross Validation - Adam Solver",
    )
    cross_validation_modes.add(healthy_vs_kc_vs_cly_cross_validation_adam)
    



    healthy_vs_kc_vs_cly_cross_validation_adam_no_bad_kcs = Mode ( # TODO 
        mode=[modes_dir, "healthy_vs_kc_vs_cly_cross_validation_adam_no_bad_kcs"],
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("cly", label_c)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        save_snapshots=False,
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_logs_on_gpu,
        dummy=False,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC vs. CLY Cross Validation - Adam Solver",
    )
    cross_validation_modes.add(healthy_vs_kc_vs_cly_cross_validation_adam)

    healthy_vs_kc_vs_cly_train_on_all_data_for_prediction_on_sus_adam = Mode(  # DONE
        mode=[modes_dir, "healthy_vs_kc_vs_cly_train_on_all_data_for_prediction_on_sus_adam"],
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("cly", label_c)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        save_snapshots=True,
        val_set_fraction=0,
        solver_net_parameters=solver_for_big_data_on_gpu,
        dummy=False,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC vs. CLY Cross Validation - Adam Solver",
    )
    cross_validation_modes.add(healthy_vs_kc_vs_cly_train_on_all_data_for_prediction_on_sus_adam)
    

    healthy_vs_kc_vs_cly_cross_validation_adam_100_epochs = Mode ( # DONE
        mode=[modes_dir, "healthy_vs_kc_vs_cly_cross_validation_adam_100_epochs"],
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("cly", label_c)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        save_snapshots=False,
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_big_data,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. KC vs. CLY Cross Validation - Adam Solver",
    )
    cross_validation_modes.add(healthy_vs_kc_vs_cly_cross_validation_adam_100_epochs)

    healthy_vs_kc_vs_cly_cross_validation_every_image_augmented_twice_adam_100_epochs = Mode(  # DONE
        mode=[modes_dir, "healthy_vs_kc_vs_cly_cross_validation_every_image_augmented_twice_adam_100_epochs"],
        targets_and_labels=(("healthy", label_a), ("kc", label_b), ("cly", label_c)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        save_snapshots=False,
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_big_data,
        dummy=True,
        augmented_data=True,
        data_dir=augment_every_image_twice_path,
        plot_title="Healthy vs. KC vs. CLY Cross Validation - Image are contrast and blur augmented Adam Solver",
        )
    cross_validation_modes.add(healthy_vs_kc_vs_cly_cross_validation_every_image_augmented_twice_adam_100_epochs)
    

    healthy_vs_cly_cross_validation_adam = Mode (  
        mode=[modes_dir, "healthy_vs_cly_cross_validation_adam"],
        targets_and_labels=(("healthy", label_a),  ("cly", label_b)),
        # txts_data=Txts_data(train_healthy=1, train_kc=1,val_healthy=1, val_kc=1),
        save_snapshots=False,
        val_set_fraction=0.05,
        solver_net_parameters=adam_solver_for_big_data,
        dummy=True,
        data_dir=my_model_data,
        plot_title="Healthy vs. CLY Cross Validation - Adam Solver",
    )
    cross_validation_modes.add(healthy_vs_cly_cross_validation_adam)
    