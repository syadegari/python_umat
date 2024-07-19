from umat.simulate import main
from umat.config import Config


if __name__ == "__main__":
    cfg = Config(
        buffer_size=300_000,
        min_buffer_size=50_000,
        batch_size=512,
        #
        N_iteration=5000,
        N_validation=10,
        #
        dataset_path="./umat/data_pairs_test.h5",
        #
        n_time=400,
        #
        coeff_data=1.0,
        coeff_physics=10.0,
        penalty_coeff_delta_gamma=100.0,
        penalty_coeff_max_slipres=100.0,
        penalty_coeff_min_slipres=100.0,
        #
        lr=5e-5,
        #
        use_lr_scheduler=True,
        lr_scheduler_T_0=2000,
        lr_scheduler_T_multi=2,
        lr_scheduler_initial_lr=5e-5,
        lt_scheduler_final_lr=2.0e-6,
        lr_scheduler_eta_min=1.0e-6,
        #
        split_train_proportion=0.7,
        split_val_proportion=0.15,
        split_test_proportion=0.15,
        #
        dataset_batch_train=32,
        dataset_batch_val=4,
        dataset_batch_test=4,
        #
        log_directory="logs3",
    )

    main(cfg)
