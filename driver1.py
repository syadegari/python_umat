from umat.simulate import main
from umat.config import Config


if __name__ == "__main__":
    cfg = Config(
        buffer_size=20000,
        min_buffer_size=10000,
        batch_size=256,
        #
        N_iteration=1000,
        N_validation=10,
        #
        dataset_path="./umat/data_pairs_test.h5",
        #
        n_time=400,
        #
        coeff_data=1.0,
        coeff_physics=1.0,
        penalty_coeff_delta_gamma=100.0,
        penalty_coeff_max_slipres=100.0,
        penalty_coeff_min_slipres=100.0,
        #
        lr=1e-4,
        #
        split_train_proportion=0.7,
        split_val_proportion=0.15,
        split_test_proportion=0.15,
        #
        dataset_batch_train=64,
        dataset_batch_val=4,
        dataset_batch_test=4,
    )

    main(cfg)
