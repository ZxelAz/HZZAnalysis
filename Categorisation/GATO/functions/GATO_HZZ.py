import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from gatohep.losses import high_bkg_uncertainty_penalty, low_bkg_penalty
from gatohep.models import gato_gmm_model
from gatohep.plotting_utils import (
    assign_bins_and_order,
    fill_histogram_from_assignments,
    plot_bias_history,
    plot_bin_boundaries_2D,
    plot_history,
    plot_learned_gaussians,
    plot_significance_comparison,
    plot_stacked_histograms,
    plot_yield_vs_uncertainty,
    make_gif
)
from gatohep.utils import (
    LearningRateScheduler,
    TemperatureScheduler,
    asymptotic_significance,
    compute_significance_from_hists,
    create_hist
)

tfd = tfp.distributions

# STXS Stage 1.2 with GG2H_GE2J_MJJ_GT350 and QQ2HQQ_rest  merged
STXS_STAGE_1_2_DICT_MERGED = {
    0: 'UNKNOWN',
    # Gluon fusion
    100: 'GG2H_FWDH',
    101: 'GG2H_PTH_GT200', # 101, 102, 103, 104 merged into one category
    105: 'GG2H_0J_PTH_0_10',
    106: 'GG2H_0J_PTH_GT10',
    107: 'GG2H_1J_PTH_0_60',
    108: 'GG2H_1J_PTH_60_120',
    109: 'GG2H_1J_PTH_120_200',
    110: 'GG2H_GE2J_MJJ_0_350_PTH_0_60',
    111: 'GG2H_GE2J_MJJ_0_350_PTH_60_120',
    112: 'GG2H_GE2J_MJJ_0_350_PTH_120_200',
    113: 'GG2H_GE2J_MJJ_GT350', # 113, 114, 115, 116 merged into one category
    # VBF
    200: 'QQ2HQQ_FWDH',
    202: 'QQ2HQQ_rest', # 201,202,203,205 merged into one category
    204: 'QQ2HQQ_GE2J_MJJ_60_120', 
    206: 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200',
    207: 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
    208: 'QQ2HQQ_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25', # 208, 210 merged into one category
    209: 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
    # qq -> WH
    300: 'QQ2HLNU_FWDH',
    301: 'VH_lep_PTV_0_150', # 301, 302, 401, 402, 501, 502 merged into one category
    303: 'VH_lep_PTV_GT150', # 303, 304, 305, 403, 404, 405, 503, 504, 505 merged into one category
    # qq -> ZH
    400: 'QQ2HLL_FWDH',
    # gg -> ZH
    500: 'GG2HLL_FWDH',
    # ttH
    600: 'TTH_FWDH',
    601: 'TTH', # 601, 602, 603, 604, 605 merged into one category
}

def format_data(proba, weights, y, class_names):
    data_dict = {name: {"NN_output": [], "weight": []} for name in class_names}
    unique_labels = np.unique(y)
    for label in unique_labels:
        mask = y == label
        p = proba[mask]
        w = weights[mask]
        for p_i, w_i in zip(p, w):
            data_dict[class_names[label]]["NN_output"].append(p_i)
            data_dict[class_names[label]]["weight"].append(w_i)
            
    return data_dict

def convert_data_to_tensors(data):
    tensor_data = {}
    for proc, content in data.items():
        # Handle both list and DataFrame formats
        if isinstance(content["NN_output"], list):
            nn = np.stack(content["NN_output"])
            w = np.array(content["weight"])
        else:
            nn = np.stack(content["NN_output"].values)
            w = content["weight"].values
        
        tensor_data[proc] = {
            "NN_output": tf.constant(nn, dtype=tf.float32),
            "weight": tf.constant(w, dtype=tf.float32),
        }
    return tensor_data



class gato_BDT(gato_gmm_model):
    def __init__(self, n_cats, dim, temperature=0.3, name="gato_BDT"):
        super().__init__(
            n_cats=n_cats,
            dim=dim,
            temperature=temperature,
            mean_norm="softmax",
            cov_offdiag_damping=0.1,
            name=name
        )

    def call(self, data_dict):
        """
            Compute the training loss and background yields,
            which can be used for penalty terms.
        """
        # Get all signal labels (everything except UNKNOWN)
        signal_labels = [proc for proc in data_dict.keys() if proc != "UNKNOWN"]
        
        significances, bkg_yield, bkg_sum_w2 = self.get_differentiable_significance(
            data_dict,
            signal_labels=signal_labels,
            return_details=True,
        )
        
        # Compute geometric mean of all signal significances
        Z_values = [significances[sig] for sig in signal_labels]
        Z_product = tf.reduce_prod(tf.stack(Z_values))
        loss = -tf.pow(Z_product, 1.0 / len(signal_labels))
        
        return loss, bkg_yield, bkg_sum_w2


# Main function to run the example
def main():
    parser = argparse.ArgumentParser(description="GATO optimisation for the BDT output of the HZZ STXS classification.")
    parser.add_argument("--data-path",   type=str, required=True, help="Path to the input data (np file with proba, weights, y, class_names).")
    parser.add_argument("--epochs",      type=int,   default=250)
    parser.add_argument("--gato-bins",   nargs="+",  type=int, default=[25])
    parser.add_argument("--lam-yield",   type=float, default=0.)
    parser.add_argument("--lam-unc",     type=float, default=0.)
    parser.add_argument("--thr-yield",   type=float, default=5.)
    parser.add_argument("--thr-unc",     type=float, default=0.20)
    parser.add_argument("--out",         type=str,   default="Plots")
    args = parser.parse_args()

    path_plots = f'./HZZ_analysis/trained_models/{args.out}/'
    os.makedirs(path_plots, exist_ok=True)

    # load the BDT output data
    print(f"Loading data from {args.data_path}...")
    data_file = np.load(args.data_path)
    
    # Extract training data (you can also use val or test)
    proba_train = data_file['proba_train']
    y_train = data_file['y_train']
    ew_train = data_file['ew_train']
    class_names_raw = data_file['class_names'].tolist() if isinstance(data_file['class_names'], np.ndarray) else list(data_file['class_names'])
    
    # Map class IDs to human-readable names using STXS dictionary
    class_names = [STXS_STAGE_1_2_DICT_MERGED.get(cls, str(cls)) for cls in class_names_raw]
    
    print(f"Loaded {len(proba_train)} training samples")
    print(f"Class names: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    # Format data into the structure expected by GATO
    data = format_data(proba_train, ew_train, y_train, class_names)

    # Get signal processes (everything except UNKNOWN)
    signal_processes = [p for p in class_names if p != "UNKNOWN"]
    print(f"Signal processes: {signal_processes}")

    # argmax-classification as comparison (using numpy)
    print("\nComputing argmax-classification baseline...")
    baseline_results = {sig: {} for sig in signal_processes}
    
    # Get argmax predictions from BDT probabilities
    y_pred = np.argmax(proba_train, axis=1)
    
    # Collect Z-scores for geometric mean calculation
    Z_scores = []
    
    # For each signal process
    for sig_idx, sig_name in enumerate(signal_processes):
        signal_sum = 0.0
        bg_sum = 0.0
        
        # Loop through predicted categories
        for pred_cat in np.unique(y_pred):
            mask = y_pred == pred_cat
            cat_weights = ew_train[mask]
            cat_labels = y_train[mask]
            
            # True signal mask (where true class is this signal)
            sig_mask = cat_labels == sig_idx
            bkg_mask = ~sig_mask
            
            signal_sum += np.sum(cat_weights[sig_mask])
            bg_sum += np.sum(cat_weights[bkg_mask])
        
        # Compute significance using numpy (Asimov formula)
        eps = 1e-10
        safe_B = np.maximum(bg_sum, eps)
        ratio = signal_sum / safe_B
        Z = np.sqrt(2.0 * ((signal_sum + safe_B) * np.log(1.0 + ratio) - signal_sum))
        
        baseline_results[sig_name][len(baseline_results[sig_name])] = float(Z)
        Z_scores.append(float(Z))
        print(f"  {sig_name}: Z = {Z:.3f}")
    
    # Compute geometric mean of Z-scores
    Z_product = np.prod(Z_scores)
    Z_geom_mean = np.power(Z_product, 1.0 / len(Z_scores))
    print(f"\nGeometric mean of Z-scores: {Z_geom_mean:.3f}")
    
        
        
        

    # GATO optimisation below
    print("\nStarting GATO optimization...")
    gato_results = {sig: {} for sig in signal_processes}
    path_gato = os.path.join(path_plots, "gato")
    os.makedirs(path_gato, exist_ok=True)

    # Create dictionary of DataFrames for validation and plotting
    data_np = {}
    for proc, content in data.items():
        nn_output_list = [arr for arr in content["NN_output"]]
        data_np[proc] = pd.DataFrame({
            "NN_output": nn_output_list,
            "weight": content["weight"]
        })
    
    # Determine number of dimensions from the data
    first_proc = list(data_np.keys())[0]
    first_nn_output = data_np[first_proc]["NN_output"].iloc[0]
    n_dims = len(first_nn_output)
    print(f"Using {n_dims} dimensions for GATO optimization")

    # Build a subsampled eval dataset to avoid OOM during per-event covariance
    # computations in get_bias / assign_bins_and_order / compute_hard_bkg_stats.
    # Build a subsampled training dataset for the same reason: the backward pass
    # through the GMM creates [N, n_cats, dim, dim] intermediate tensors that
    # overflow CUDA int32 and exhaust GPU memory when N is large.
    # Weights are rescaled so the total weight per process is preserved,
    # keeping the significance loss function unbiased.
    TRAIN_MAX_PER_PROC = 50_000
    EVAL_MAX_PER_PROC  = 10_000
    rng = np.random.default_rng(seed=42)

    def _subsample(content, max_n):
        nn_arr = np.stack(content["NN_output"])
        w_arr  = np.array(content["weight"])
        n = len(w_arr)
        if n > max_n:
            idx   = rng.choice(n, size=max_n, replace=False)
            scale = n / max_n          # preserve total weight
            nn_arr = nn_arr[idx]
            w_arr  = w_arr[idx] * scale
        return nn_arr, w_arr

    tensor_data = {}
    eval_data   = {}
    for proc, content in data.items():
        nn_tr, w_tr   = _subsample(content, TRAIN_MAX_PER_PROC)
        nn_ev, w_ev   = _subsample(content, EVAL_MAX_PER_PROC)
        tensor_data[proc] = {
            "NN_output": tf.constant(nn_tr, dtype=tf.float32),
            "weight":    tf.constant(w_tr,  dtype=tf.float32),
        }
        eval_data[proc] = {
            "NN_output": tf.constant(nn_ev, dtype=tf.float32),
            "weight":    tf.constant(w_ev,  dtype=tf.float32),
        }

    eval_data_np = {}
    for proc, content in eval_data.items():
        nn_list = [content["NN_output"][i].numpy() for i in range(len(content["weight"]))]
        eval_data_np[proc] = pd.DataFrame({
            "NN_output": nn_list,
            "weight":    content["weight"].numpy(),
        })

    total_train = sum(len(v["weight"]) for v in tensor_data.values())
    print(f"Train subset: up to {TRAIN_MAX_PER_PROC} events per process "
          f"({total_train} total, weights rescaled)")
    print(f"Eval  subset: up to {EVAL_MAX_PER_PROC} events per process")

    for n_cats in args.gato_bins:

        @tf.function
        def train_step(model, tdata, opt, lamY, lamU, thrY, thrU):
            with tf.GradientTape() as tape:
                loss, B, Bw2 = model.call(tdata)
                penY = low_bkg_penalty(B, threshold=thrY)
                penU = high_bkg_uncertainty_penalty(Bw2, B, rel_threshold=thrU)
                total = loss + lamY*penY + lamU*penU
            g = tape.gradient(total, model.trainable_variables)
            opt.apply_gradients(zip(g, model.trainable_variables))
            return loss

        model = gato_BDT(n_cats=n_cats, dim=n_dims, temperature=1.0)

        optimizer = tf.keras.optimizers.RMSprop(0.1)
        lr_scheduler = LearningRateScheduler(
            optimizer,
            lr_initial=0.05,
            lr_final=0.001,
            total_epochs=args.epochs,
            mode="cosine",
        )

        # temperature scheduler
        temperature_scheduler = TemperatureScheduler(
            model,
            t_initial=1.0,
            t_final=0.05,
            total_epochs=args.epochs,
            mode="cosine",
        )

        loss_history = []
        boundary_frames = []
        hist_frames = []
        mean_bias_history = []
        bias_epochs = []
        temperature_history = []
        for ep in range(args.epochs):
            lr_scheduler.update(ep)
            temperature_scheduler.update(ep)
            loss = train_step(
                model, tensor_data, optimizer,
                args.lam_yield, args.lam_unc,
                args.thr_yield, args.thr_unc
            )

            if ep % 25 == 0 or ep == args.epochs - 1:
                bias_vec = model.get_bias(eval_data)
                mean_bias_history.append(float(np.mean(np.abs(bias_vec))))
                bias_epochs.append(ep)
                temperature_history.append(float(model.temperature))

            if ep % 25 == 0:
                lr_value = getattr(optimizer, "learning_rate", getattr(optimizer, "lr", None))
                if hasattr(lr_value, "numpy"):
                    lr_value = float(lr_value.numpy())
                else:
                    lr_value = float(lr_value)
                print(f"[{ep:03d}] loss = {loss.numpy():.3f}, lr = {lr_value:.5f}")
                assign, order, _, inv = assign_bins_and_order(
                    model, eval_data_np, reduce=True
                )
                filled = {p: fill_histogram_from_assignments(
                    assign[p], eval_data_np[p]["weight"], n_cats
                ) for p in eval_data_np}
                # Background is the "UNKNOWN" category
                bg_procs = ["UNKNOWN"] if "UNKNOWN" in eval_data_np else []
                opt_bkgs = [filled[p] for p in bg_procs]
                
                # Create signal histograms (scaled for visibility)
                signal_hists = [100 * filled[sig] for sig in signal_processes]
                signal_labels = [f"{sig} x100" for sig in signal_processes]
                
                # 1) histogram
                hist_fname = path_gato + f"/progress_plots_{n_cats}/hist_{ep:04d}.png"
                plot_stacked_histograms(
                    stacked_hists=opt_bkgs,
                    process_labels=bg_procs,
                    signal_hists=signal_hists,
                    signal_labels=signal_labels,
                    output_filename=hist_fname,
                    axis_labels=("Bin index", "Events"),
                    normalize=False,
                    log=True,
                )
                hist_frames.append(hist_fname)

                # 2) boundaries (only for 2D models)
                if n_dims == 2:
                    boundary_fname = path_gato + f"/frames_{n_cats}/boundary_{ep:04d}.png"
                    plot_bin_boundaries_2D(
                        model,
                        [i for i in range(n_cats)],
                        boundary_fname,
                        resolution=500,
                        annotation=f"Epoch {ep}",
                    )
                    boundary_frames.append(boundary_fname)
            loss_history.append(loss.numpy())

        checkpoint_dir = os.path.join(path_gato, "checkpoints", f"{n_cats}_bins")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save(checkpoint_dir)

        bias_plot_base = os.path.join(path_gato, f"bias_history_{n_cats}bins")
        plot_bias_history(
            mean_bias_history,
            bias_plot_base + ".pdf",
            epochs=bias_epochs,
            temp_points=temperature_history,
            temp_label="Temperature",
        )
        plot_bias_history(
            mean_bias_history,
            bias_plot_base + "_log.pdf",
            epochs=bias_epochs,
            temp_points=temperature_history,
            temp_label="Temperature",
            log_scale=True,
        )

        # check bias due to finite temperature in training
        bias = model.get_bias(eval_data)
        print(f"T = {model.temperature:4.2f};  per-bin bias: {bias}")

        assign, order, _, inv = assign_bins_and_order(model, eval_data_np, reduce=True)

        # 2) make per-process histograms
        filled = {p: fill_histogram_from_assignments(
            assign[p], eval_data_np[p]["weight"], n_cats
        ) for p in eval_data_np}

        # Compute significances for each signal
        bg_procs = [p for p in data_np if p != "UNKNOWN" and p not in signal_processes]
        opt_bkgs = [filled[p] for p in bg_procs]
        
        for sig_name in signal_processes:
            # Other signals are treated as background
            other_signals = [filled[s] for s in signal_processes if s != sig_name]
            Z = compute_significance_from_hists(
                filled[sig_name], opt_bkgs + other_signals
            )
            gato_results[sig_name][n_cats] = Z
            print(f"  {sig_name}: Z = {Z:.3f}")

        # quick plots (only for 2D models)
        if n_dims == 2:
            plot_learned_gaussians(
                data=data_np, model=model, dim_x=0, dim_y=1,
                output_filename=os.path.join(path_gato, f"Gaussians_{n_cats}bins.pdf"),
                inv_mapping=inv,
            )

            plot_bin_boundaries_2D(
                model,
                order,
                path_plot=os.path.join(path_gato, f"Bin_boundaries_{n_cats}_bins.pdf")
            )

        # loss curve
        plot_history(
            np.array(loss_history),
            os.path.join(path_gato, f"loss_{n_cats}.pdf"),
            y_label=r"Geometric mean of signal Z-scores", x_label="Epoch"
        )

        # 1) Stacked histogram of optimized bins:
        # Collect background processes (UNKNOWN is the background)
        bg_procs = ["UNKNOWN"] if "UNKNOWN" in filled else []
        opt_bkgs = [filled[p] for p in bg_procs]
        
        # Create signal histograms (scaled for visibility)
        signal_hists = [100 * filled[sig] for sig in signal_processes]
        signal_labels = [f"{sig} x100" for sig in signal_processes]
        print(f"DEBUG FINAL: Number of signals: {len(signal_hists)}, signal_processes: {len(signal_processes)}")

        for use_log in (False, True):
            suffix = "log" if use_log else "linear"
            
            # Only plot if we have histograms to show
            if opt_bkgs or signal_hists:
                plot_stacked_histograms(
                    stacked_hists=opt_bkgs,
                    process_labels=bg_procs,
                    signal_hists=signal_hists,
                    signal_labels=signal_labels,
                    output_filename=os.path.join(
                        path_gato, f"optimized_dist_{n_cats}bins_{suffix}.pdf"
                    ),
                    axis_labels=("Bin index", "Events"),
                    normalize=False,
                    log=use_log
                )
                print(
                    f"Saved optimized {suffix} histogram:\
                    optimized_dist_{n_cats}bins_{suffix}.pdf"
                )

        B_sorted, rel_unc_sorted, _ = model.compute_hard_bkg_stats(eval_data)
        B_ord = B_sorted[order]
        unc_ord = rel_unc_sorted[order]

        for use_log in (False, True):
            suffix = "log" if use_log else "linear"

            plot_yield_vs_uncertainty(
                B_ord,
                unc_ord,
                log=use_log,
                output_filename=os.path.join(
                    path_gato, f"yield_vs_unc_{n_cats}bins_{suffix}.pdf"
                )
            )
            print(
                f"Saved yield vs. unc ({suffix}):\
                yield_vs_unc_{n_cats}bins_{suffix}.pdf"
            )

        # Create frames directory before saving GIFs
        frames_dir = os.path.join(path_gato, f"frames_{n_cats}")
        os.makedirs(frames_dir, exist_ok=True)
        
        make_gif(
            hist_frames, os.path.join(frames_dir, "hist_evolution.gif")
        )
        if n_dims == 2 and boundary_frames:
            make_gif(
                boundary_frames, os.path.join(frames_dir, "boundaries_evolution.gif")
            )

    # summary comparison
    print("\nGenerating significance comparison plot...")
    
    # Reformat baseline_results for comparison plot
    baseline_reformatted = {}
    for sig in signal_processes:
        if sig in baseline_results and baseline_results[sig]:
            baseline_reformatted[sig] = {
                2*n+1: baseline_results[sig][n] for n in baseline_results[sig]
            }
    
    plot_significance_comparison(
        baseline_results=baseline_reformatted,
        optimized_results=gato_results,
        output_filename=os.path.join(path_gato, "significance_comparison.pdf"),
    )
    
    print(f"\n✅ GATO optimization completed!")
    print(f"Results saved to: {path_gato}")


if __name__ == "__main__":
    main()
