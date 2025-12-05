#from src.model_classification import run
from src.model_regression import run_opt_reg
from src.model_classification import run_opt_class

def multiple_opt_reg():
    feature_types = ["fft_cpmg", "cpmg", "echo_sample"]
    target_types = ["mqi"l]

    for feature_type in feature_types:
        for target_type in target_types:
            feature_method = "raw"
            use_pca = False
            use_feature_engineering = True
            print(f"Running optimization with: "
                    f"feature_type={feature_type}, "
                    f"target_type={target_type}, "
                    f"feature_method={feature_method}, "
                    f"use_pca={use_pca}, "
                    f"use_feature_engineering={use_feature_engineering}")
            run_opt_reg(
                feature_type=feature_type,
                target_type=target_type,
                feature_method=feature_method,
                use_pca=use_pca,
                use_feature_engineering=use_feature_engineering,
                n_trials=100,
                plot_suffix=f"_{feature_type}_{target_type}"
            )

def multiple_opt_class():
    feature_types = ["fft_cpmg", "cpmg", "echo_sample"]
    target_types = ["mqi"]

    for feature_type in feature_types:
        for target_type in target_types:
            feature_method = "raw"
            use_pca = False
            use_feature_engineering = True
            print(f"Running optimization with: "
                    f"feature_type={feature_type}, "
                    f"target_type={target_type}, "
                    f"feature_method={feature_method}, "
                    f"use_pca={use_pca}, "
                    f"use_feature_engineering={use_feature_engineering}")
            run_opt_class(
                feature_type=feature_type,
                target_type=target_type,
                feature_method=feature_method,
                use_pca=use_pca,
                use_feature_engineering=use_feature_engineering,
                n_trials=5,
                plot_suffix=f"_{feature_type}_{target_type}"
            )

if __name__ == "__main__":

    multiple_opt_reg()
    multiple_opt_class()