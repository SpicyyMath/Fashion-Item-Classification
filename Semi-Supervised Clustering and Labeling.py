# -*- coding: utf-8 -*-
# Add this at the top if you have non-ASCII characters in comments or strings

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import torch # Needed for Tensor checking and manipulation
import pandas as pd # For feature distribution plotting if added later
import seaborn as sns # For feature distribution plotting if added later
from sklearn.svm import SVC # Still needed if Classifier wraps it, but Classifier itself is imported

# --- Try importing essential ML libraries ---
try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.ensemble import RandomForestClassifier # For feature importance analysis
    from sklearn.cluster import DBSCAN # Import DBSCAN here as it's now the default/only clusterer used
except ImportError:
    print("FATAL ERROR: Essential library scikit-learn not installed.")
    print("Please install it ('pip install scikit-learn') for scaling, PCA, NN, RF, DBSCAN, and cluster visualization.")
    # Exit or re-raise to stop execution is appropriate here, but for now just print.
    TSNE, StandardScaler, PCA, NearestNeighbors, RandomForestClassifier, DBSCAN = None, None, None, None, None, None
    # Depending on structure, might want to exit() here.

# --- Import Project Modules ---
# These imports are now REQUIRED. If they fail, the script will stop with an ImportError.
try:
    from utils.data_loader_n import FashionSuperclassDataset
    from models.cluster_model import FeatureExtractor
    # Import the Classifier from the model definition - THIS MUST EXIST AND BE CORRECT
    from models.cluster_model import Classifier
    print("Successfully imported required project modules: FashionSuperclassDataset, FeatureExtractor, Classifier.")

except ImportError as e:
    print(f"\nFATAL ERROR: Could not import required project modules: {e}")
    print("Please ensure 'utils/data_loader_n.py' and 'models/cluster_model.py' exist")
    print("and contain the necessary classes (FashionSuperclassDataset, FeatureExtractor, Classifier).")
    print("Ensure the Classifier in 'models/cluster_model.py' accepts 'kernel' and 'C' arguments in __init__.")
    raise e # Stop execution by re-raising the import error

# --- Parameters ---
IMAGE_DIR = "Data/images"
LABEL_PATH = "Data/labels.json"
# --- TUNABLE PARAMETERS ---
APPLY_PCA = True
# >>> TUNE THIS: Consider variance explained. Larger values = more variance explained.
PCA_N_COMPONENTS = 32 
# >>> TUNE THIS: CRITICAL - Adjust based on the K-distance plot generated *for the PCA_N_COMPONENTS used*. Larger = more points in a cluster.
DBSCAN_EPS = 300.0
# >>> TUNE THIS: Minimal points considered as a cluster.
DBSCAN_MIN_SAMPLES = 500
# --- CONTROL FLAGS ---
PLOT_K_DISTANCE = True # Generate K-distance plot to help choose EPS
VISUALIZE_CLUSTERS = True # Generate t-SNE plot of final clusters
ANALYZE_FEATURES = True # Perform post-clustering feature analysis
# --------------------------
SVM_KERNEL = 'rbf' # Kernel for the SVM
SVM_C = 1.0        # Regularization parameter for the SVM
NUM_SAMPLES_TO_VISUALIZE = 9
OUTPUT_FILE = 'final_labels.npy'


DATASET_MEAN = torch.tensor([0.485, 0.456, 0.406])
DATASET_STD = torch.tensor([0.229, 0.224, 0.225])

# --- Helper Function for Denormalization ---
def denormalize(tensor, mean, std):
    """Denormalizes a tensor image with mean and standard deviation."""
    mean = torch.tensor(mean) if not isinstance(mean, torch.Tensor) else mean
    std = torch.tensor(std) if not isinstance(std, torch.Tensor) else std
    tensor = tensor.clone()
    mean = mean.view(-1, 1, 1).to(tensor.device)
    std = std.view(-1, 1, 1).to(tensor.device)
    tensor.mul_(std).add_(mean)
    return tensor

# --- Main Function ---
def main():
    # Check if essential sklearn components are available after import attempt
    if StandardScaler is None or PCA is None or NearestNeighbors is None or DBSCAN is None:
         print("Error: Core scikit-learn components missing. Cannot proceed.")
         return

    print("--- Script Parameters ---")
    print(f"Image Directory: {IMAGE_DIR}"); print(f"Label Path: {LABEL_PATH}")
    print(f"Apply PCA: {APPLY_PCA}");
    if APPLY_PCA: print(f"PCA Components: {PCA_N_COMPONENTS}")
    print(f"DBSCAN eps (Value to be used): {DBSCAN_EPS}")
    print(f"DBSCAN min_samples: {DBSCAN_MIN_SAMPLES}")
    print(f"Plot K-Distance Graph: {PLOT_K_DISTANCE}"); print(f"Visualize Clusters: {VISUALIZE_CLUSTERS}"); print(f"Analyze Features: {ANALYZE_FEATURES}")
    print(f"Output File: {OUTPUT_FILE}")
    print(f"Using Mean/Std for Denorm: {DATASET_MEAN.tolist()} / {DATASET_STD.tolist()}")
    print("-" * 25)

    # --- Load Data ---
    print("Loading dataset...")
    try:
        # Use the imported FashionSuperclassDataset
        dataset = FashionSuperclassDataset(image_dir=IMAGE_DIR, label_path=LABEL_PATH)
        all_image_identifiers, has_label, actual_labels = dataset.prepare_data_lists()
        print(f"Dataset loaded. Found {len(all_image_identifiers)} items.")
        total_images = len(all_image_identifiers)
        if total_images == 0: print("Error: No items loaded from dataset. Exiting."); return
    except Exception as e: print(f"Error loading data: {e}"); return

    # --- Input Data Validation ---
    print("\n--- Input Data Validation ---")
    labeled_count=sum(has_label); unlabeled_count=total_images-labeled_count; unique_labels=set(l for l in actual_labels if l is not None)
    print(f"Total items: {total_images}"); print(f"Labeled: {labeled_count} ({labeled_count/total_images:.2%})"); print(f"Unlabeled: {unlabeled_count} ({unlabeled_count/total_images:.2%})")
    print(f"Unique known labels ({len(unique_labels)}): {', '.join(map(str, sorted(list(unique_labels))[:15]))}{'...' if len(unique_labels)>15 else ''}")
    print("-" * 25)

    # --- Visualize Data Samples ---
    print("\nVisualizing some data samples...")
    # (Visualization code remains the same)
    num_to_show=min(NUM_SAMPLES_TO_VISUALIZE,total_images)
    if num_to_show>0:
        sample_indices=random.sample(range(total_images), num_to_show)
        cols=int(np.ceil(np.sqrt(num_to_show))); rows=int(np.ceil(num_to_show/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*3))
        if isinstance(axes, plt.Axes): axes=np.array([axes]) # Ensure axes is iterable
        axes=axes.flatten(); vis_count=0
        for i, idx in enumerate(sample_indices):
            if i >= len(axes): break # Stop if we run out of axes
            if idx>=len(all_image_identifiers): continue # Basic bounds check
            img_id=all_image_identifiers[idx]; label=actual_labels[idx] if has_label[idx] else "No Label"; ax=axes[i]; ax.axis('off')
            try: # Visualization logic
                img_to_show=None
                if isinstance(img_id,torch.Tensor):
                    tensor=img_id.cpu()
                    try:
                         n_ch = tensor.shape[0]
                         current_mean = DATASET_MEAN[:n_ch] if n_ch < len(DATASET_MEAN) else DATASET_MEAN
                         current_std = DATASET_STD[:n_ch] if n_ch < len(DATASET_STD) else DATASET_STD
                         if n_ch == len(current_mean):
                             denorm_tensor=denormalize(tensor, current_mean, current_std)
                         else:
                             denorm_tensor=tensor
                    except Exception as de: print(f"E: denorm viz {idx}: {de}"); denorm_tensor=tensor
                    if denorm_tensor.ndim==3 and denorm_tensor.shape[0] in [1,3]:
                         display_tensor=denorm_tensor.permute(1,2,0)
                    elif denorm_tensor.ndim==2:
                         display_tensor=denorm_tensor
                    else:
                         display_tensor=None

                    if display_tensor is not None:
                        np_img=display_tensor.detach().numpy()
                        if np_img.dtype == np.float32 or np_img.dtype == np.float64: np_img = np.clip(np_img, 0, 1)
                        elif np_img.dtype == np.uint8: np_img = np.clip(np_img, 0, 255)
                        if np_img.ndim==3 and np_img.shape[2]==1: np_img=np_img.squeeze(axis=2)
                        img_to_show=np_img

                elif isinstance(img_id,str) and os.path.exists(img_id): img_to_show=Image.open(img_id).convert('RGB')
                elif isinstance(img_id,Image.Image): img_to_show=img_id.convert('RGB')
                else: print(f"W: Cannot display sample {idx} type {type(img_id)}."); ax.text(0.5,0.5,'Cannot display',ha='center',va='center',fontsize=8)

                if img_to_show is not None:
                    is_gray=isinstance(img_to_show, np.ndarray) and img_to_show.ndim==2
                    ax.imshow(img_to_show, cmap='gray' if is_gray else None); ax.set_title(f"Idx: {idx}\nL: {label}",fontsize=8); vis_count+=1
                else:
                     ax.text(0.5,0.5,'Cannot display',ha='center',va='center',fontsize=8)
            except Exception as e: print(f"E: display img {idx}: {e}"); ax.text(0.5,0.5,'Error display',ha='center',va='center',fontsize=8)
        for j in range(vis_count, len(axes)): axes[j].axis('off')
        plt.suptitle(f"Random Samples ({vis_count}/{num_to_show})", fontsize=12); plt.tight_layout(rect=[0,0.03,1,0.95]); plt.show()
    else: print("No samples to visualize.")
    print("-" * 25)


    # --- Extract Features ---
    print("\nExtracting features for all items...")
    all_features=None # This will store the ORIGINAL high-dim features
    try:
        extractor=FeatureExtractor()
        all_features_list=[extractor.extract(item) for item in tqdm(all_image_identifiers, desc="Extracting Features", unit="item")]
        if not all_features_list: print("Error: No features extracted."); return
        # Store original features; ensure they are numpy arrays for potential later use if needed
        all_features = [np.array(f) for f in all_features_list]
        print(f"Feature extraction done. Example original feature shape: {all_features[0].shape}")
    except Exception as e: print(f"Error during feature extraction: {e}"); return
    print("-" * 25)


    # --- Preprocessing: Scaling & Optional PCA ---
    print("\nPreprocessing features...")
    features_final_for_clustering = None # This will store the LOW-DIM features
    features_for_tsne = None
    pca_model = None
    try:
        if not all_features: raise ValueError("Original features are missing.")
        print("Preparing features (flattening if needed)...")
        try:
             # Prepare features for scaling/PCA from the original list
             features_prepared=np.stack([f.flatten() for f in all_features])
        except ValueError as sve:
             shapes = {f.shape for f in all_features}
             raise ValueError(f"Inconsistent original feature shapes detected: {shapes}. Cannot stack. Error: {sve}")

        if features_prepared.ndim != 2: raise ValueError(f"Prepared features are not 2D. Shape: {features_prepared.shape}")
        print(f"Prepared original features shape: {features_prepared.shape}")

        print("Scaling features (mean=0, std=1)...");
        scaler=StandardScaler(); features_scaled=scaler.fit_transform(features_prepared);
        print(f"Scaling done. Shape: {features_scaled.shape}. Mean check: {features_scaled.mean():.2f}, Std check: {features_scaled.std():.2f}")

        if APPLY_PCA:
            n_comps = PCA_N_COMPONENTS
            print(f"Applying PCA -> target components: {n_comps}...")
            if isinstance(n_comps, int) and n_comps > features_scaled.shape[1]:
                print(f"W: PCA_N_COMPONENTS ({n_comps}) > n_features ({features_scaled.shape[1]}). Using n_features instead.")
                n_comps = features_scaled.shape[1]
            elif isinstance(n_comps, float) and (n_comps <= 0 or n_comps > 1):
                 raise ValueError("PCA_N_COMPONENTS must be >0 and <=1 if float (variance ratio), or a positive integer.")

            pca_model = PCA(n_components=n_comps, random_state=42)
            # This is the low-dimensional feature set
            features_final_for_clustering = pca_model.fit_transform(features_scaled)
            explained_var = np.sum(pca_model.explained_variance_ratio_)
            actual_components = features_final_for_clustering.shape[1]
            print(f"PCA done. Actual components: {actual_components}. Shape: {features_final_for_clustering.shape}.")
            print(f"--> Explained variance by {actual_components} components: {explained_var:.4f}")
            if explained_var < 0.5: print("W: Explained variance is very low (< 50%). Consider increasing PCA_N_COMPONENTS or disabling PCA.")
        else:
            print("Skipping PCA. Using scaled features for clustering.")
            features_final_for_clustering = features_scaled # Use scaled if no PCA

        # Features for t-SNE are the same low-dimensional ones used for clustering
        features_for_tsne = features_final_for_clustering
        print(f"Features ready for DBSCAN/t-SNE (LOW-DIM). Shape: {features_final_for_clustering.shape}")

    except Exception as e: print(f"Error during preprocessing: {e}"); return
    print("-" * 25)


    # --- Estimate EPS using K-distance plot (Optional) ---
    if PLOT_K_DISTANCE:
        print("\nCalculating K-distances for EPS estimation (using LOW-DIM features)...")
        # (K-distance plot code remains the same)
        if features_final_for_clustering is not None and features_final_for_clustering.shape[0] > DBSCAN_MIN_SAMPLES:
            try:
                k_nn = DBSCAN_MIN_SAMPLES
                print(f"Finding distance to {k_nn}-th neighbor for {features_final_for_clustering.shape[0]} points...")
                nbrs = NearestNeighbors(n_neighbors=k_nn + 1, algorithm='auto', n_jobs=-1).fit(features_final_for_clustering)
                distances, indices = nbrs.kneighbors(features_final_for_clustering)
                k_distances = np.sort(distances[:, k_nn], axis=0)

                plt.figure(figsize=(10, 6)); plt.plot(k_distances)
                plt.title(f'K-Distance Graph (k={k_nn}) for {features_final_for_clustering.shape[1]}D Features')
                plt.xlabel("Points sorted by distance"); plt.ylabel(f'{k_nn}-th Nearest Neighbor Distance')
                plt.grid(True, linestyle='--')
                plt.text(0.05, 0.95,
                         f'Look for the "elbow" point where the curve bends upwards sharply.\n'
                         f'The Y-value (distance) at the elbow is a good starting estimate for DBSCAN_EPS.\n'
                         f'(Current DBSCAN_EPS is set to: {DBSCAN_EPS})',
                         transform=plt.gca().transAxes, fontsize=9, va='top',
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

                print(f"===> Displaying K-distance plot. IMPORTANT: Examine the 'elbow' and adjust DBSCAN_EPS ({DBSCAN_EPS}) if needed before proceeding. Close plot window to continue...")
                plt.show()

            except Exception as kd_e: print(f"Error in K-distance calculation/plot: {kd_e}")
        else: print("Skipping K-plot: Not enough data points or low-dim features missing.")
        print("-" * 25)


    # --- DBSCAN Clustering ---
    print("\nRunning DBSCAN clustering (using LOW-DIM features)...")
    cluster_labels = None
    try:
        if features_final_for_clustering is None: raise ValueError("Low-dimensional features for clustering are missing.")
        if features_final_for_clustering.shape[0] == 0: raise ValueError("Low-dimensional feature array for clustering is empty.")

        print(f"Using sklearn.cluster.DBSCAN on feature shape {features_final_for_clustering.shape} with eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}...")
        dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
        cluster_labels = dbscan.fit_predict(features_final_for_clustering)

        # Analyze clustering results
        unique_labels_found = set(cluster_labels)
        num_clusters = len(unique_labels_found) - (1 if -1 in unique_labels_found else 0)
        num_noise = np.sum(np.array(cluster_labels) == -1)
        print(f"Clustering done. Found {num_clusters} clusters and {num_noise} noise points ({num_noise/total_images:.2%}).")
        # (Warnings remain the same)
        if num_clusters == 0 and num_noise == total_images: print("W: ALL points classified as noise! Adjust DBSCAN_EPS/PCA.")
        elif num_clusters == 1 and num_noise < total_images * 0.1: print("W: Found only one large cluster. Adjust DBSCAN_EPS.")
        elif num_clusters > 50: print(f"W: Found a large number of clusters ({num_clusters}). Check DBSCAN_EPS.")

    except Exception as e: print(f"Error during DBSCAN clustering step: {e}"); return
    print("-" * 25)


    # --- Visualize Clusters using t-SNE ---
    if VISUALIZE_CLUSTERS:
        print("\nPreparing cluster visualization using t-SNE (from LOW-DIM features)...")
        # (t-SNE visualization code remains the same)
        if TSNE is None: print("Skipping visualization: TSNE not available.")
        elif features_for_tsne is not None and features_for_tsne.shape[0] > 1 and cluster_labels is not None:
            if features_for_tsne.shape[0] != len(cluster_labels):
                print(f"Error: Feature count ({features_for_tsne.shape[0]}) and label count ({len(cluster_labels)}) mismatch.")
            else:
                try:
                    n_samples_tsne = features_for_tsne.shape[0]
                    perplexity_val = min(30, n_samples_tsne - 1)
                    if perplexity_val <= 0: raise ValueError("Perplexity must be positive.")

                    print(f"Running t-SNE on {n_samples_tsne} samples (shape: {features_for_tsne.shape}) with perplexity={perplexity_val}...")
                    init_mode = 'pca' if features_for_tsne.shape[1] > 50 else 'random'
                    tsne = TSNE(n_components=2, perplexity=perplexity_val, learning_rate='auto',
                                init=init_mode, n_iter=1000,
                                random_state=42, n_jobs=-1)
                    features_2d = tsne.fit_transform(features_for_tsne)
                    print("t-SNE dimensionality reduction complete.")

                    plt.figure(figsize=(12, 10))
                    unique_ids = sorted(list(set(cluster_labels)))
                    num_actual_clusters = len(unique_ids) - (1 if -1 in unique_ids else 0)

                    colors = plt.cm.viridis(np.linspace(0, 0.9, max(1, num_actual_clusters)))
                    noise_color = 'lightgray'
                    plotted_labels = set()

                    for cid in unique_ids:
                         mask = (cluster_labels == cid)
                         count = np.sum(mask)
                         if count == 0: continue
                         label_text = f'Cluster {cid} ({count} pts)' if cid != -1 else f'Noise ({count} pts)'
                         color = noise_color if cid == -1 else colors[cid % len(colors)]
                         size = 10 if cid == -1 else 20; alpha = 0.5 if cid == -1 else 0.8
                         if label_text not in plotted_labels:
                              plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[color], label=label_text, s=size, alpha=alpha)
                              plotted_labels.add(label_text)
                         else:
                              plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[color], s=size, alpha=alpha)

                    pca_status = f"PCA ({features_for_tsne.shape[1]} comps)" if APPLY_PCA else "No PCA"
                    plt.title(f't-SNE Visualization ({num_actual_clusters} clusters + Noise) - Features: {pca_status}');
                    plt.xlabel('t-SNE Component 1'); plt.ylabel('t-SNE Component 2')

                    num_legend_entries = len(plotted_labels)
                    if num_legend_entries > 15:
                        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), markerscale=1.5, title="Clusters/Noise", fontsize='small')
                        plt.tight_layout(rect=[0, 0, 0.88, 1])
                    elif num_legend_entries > 0:
                        plt.legend(loc='best', markerscale=1.5); plt.tight_layout()
                    else: plt.tight_layout()
                    plt.grid(True, linestyle='--', alpha=0.5); print("Displaying cluster plot..."); plt.show()

                except Exception as viz_e: print(f"Error during cluster visualization: {viz_e}")
        else: print("Skipping visualization: Data for t-SNE is missing or labels not generated.")
    print("-" * 25)


    # --- Analyze Feature Differences (Optional) ---
    if ANALYZE_FEATURES and cluster_labels is not None and features_final_for_clustering is not None:
        print("\nAnalyzing feature statistics per cluster (using LOW-DIM features)...")
        # (Feature analysis code remains the same - it operates on features_final_for_clustering)
        try:
            unique_ids = sorted(list(set(cluster_labels)))
            num_feats = features_final_for_clustering.shape[1]
            cluster_stats = {}
            print(f"Analyzing {num_feats} features used for clustering.")

            for cid in unique_ids:
                mask = (cluster_labels == cid)
                features_in_cluster = features_final_for_clustering[mask, :]
                num_points = features_in_cluster.shape[0]
                if num_points > 0:
                     mean_f=np.mean(features_in_cluster,axis=0); std_f=np.std(features_in_cluster,axis=0)
                     cluster_stats[cid]={'mean':mean_f, 'std':std_f, 'count': num_points}
                     label_type = "Noise" if cid == -1 else "Cluster"; print(f"  - {label_type} {cid} ({num_points} pts): Mean feat val={np.mean(mean_f):.4f}, Avg Std Dev={np.mean(std_f):.4f}")
                else: print(f"  - Cluster {cid}: 0 points found.")

            non_noise_clusters = {k: v for k, v in cluster_stats.items() if k != -1}
            if len(non_noise_clusters) >= 2:
                 largest_cid = max(non_noise_clusters, key=lambda k: non_noise_clusters[k]['count'])
                 other_cids = sorted([k for k in non_noise_clusters if k != largest_cid])
                 if other_cids:
                     other_cid = other_cids[0]
                     mean_diff=np.abs(cluster_stats[largest_cid]['mean']-cluster_stats[other_cid]['mean']); top_diff_idx=np.argsort(mean_diff)[::-1]
                     print(f"\nTop 5 feature differences (mean abs val) between Largest Cluster ({largest_cid}) and Cluster {other_cid}:")
                     for i in range(min(5,len(top_diff_idx))): idx=top_diff_idx[i]; print(f"  - Feature Index: {idx}, Diff: {mean_diff[idx]:.4f}")
        except Exception as fe_e: print(f"Error during feature statistics analysis: {fe_e}")

        # --- Feature Importance using RandomForest (Optional) ---
        print("\nCalculating feature importance using RandomForest (on LOW-DIM features)...")
        try:
            if RandomForestClassifier and len(set(cluster_labels[cluster_labels != -1])) > 1:
                 non_noise_mask=(cluster_labels!=-1); X_train=features_final_for_clustering[non_noise_mask,:]; y_train=cluster_labels[non_noise_mask]
                 if X_train.shape[0] > 1 and len(set(y_train)) > 1:
                    print(f"Training RandomForest on {X_train.shape[0]} non-noise points across {len(set(y_train))} clusters...")
                    rf=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1,max_depth=10,class_weight='balanced'); rf.fit(X_train, y_train)
                    importances=rf.feature_importances_; indices=np.argsort(importances)[::-1]
                    n_features_to_show=min(10, X_train.shape[1])
                    print(f"\nTop {n_features_to_show} important features (out of {X_train.shape[1]}) for distinguishing DBSCAN clusters:")
                    for i in range(n_features_to_show): feat_idx=indices[i]; print(f"  - Feature Index: {feat_idx}, Importance: {importances[feat_idx]:.6f}")
                 else: print("Skipping RF: Not enough data or only one non-noise cluster after filtering.")
            else: print("Skipping RF: Need multiple non-noise clusters or RandomForestClassifier unavailable.")
        except Exception as rf_e: print(f"Error during RandomForest feature importance calculation: {rf_e}")
        print("-" * 25)

    # --- Group Images by Cluster ---
    print("\nGrouping items by cluster...")
    # (Grouping and distance calculation code remains the same)
    clusters = defaultdict(list)
    if cluster_labels is not None:
        for idx, lbl in enumerate(cluster_labels): clusters[int(lbl)].append(idx)
        print(f"Grouped items into {len(clusters)} groups (including noise cluster -1 if present).")
        print("\nCluster sizes:")
        total_clustered = 0; cluster_ids_sorted = sorted(clusters.keys())
        for cid in cluster_ids_sorted:
            inds = clusters[cid]; count = len(inds); label_type = "Noise" if cid == -1 else "Cluster"
            print(f"  - {label_type} {cid}: {count} points");
            if cid != -1: total_clustered += count
        print(f"Total points in actual clusters (excluding noise): {total_clustered}")
        print("\nCalculating Inter-cluster distances (Euclidean between centroids using LOW-DIM features)...")
        centroids = {}
        for cid, inds in clusters.items():
            if cid == -1 or len(inds) == 0: continue
            pts = features_final_for_clustering[inds]; centroids[cid] = np.mean(pts, axis=0)
        cluster_ids = sorted(centroids.keys())
        if len(cluster_ids) >= 2:
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    c1_id, c2_id = cluster_ids[i], cluster_ids[j]
                    if c1_id in centroids and c2_id in centroids:
                        dist = np.linalg.norm(centroids[c1_id] - centroids[c2_id])
                        print(f"  - Distance between Cluster {c1_id} and {c2_id}: {dist:.4f}")
            print("Distances based on the feature space used for clustering.")
        else: print("Not enough non-noise clusters (need >= 2) to calculate inter-cluster distances.")
    else: print("Skipping grouping and distance calculation: cluster_labels not available.")
    print("-" * 25)


# --- Train SVM per Cluster ---
    print("\nTraining SVM for each non-noise cluster with labeled data...")
    print(f"Using SVM Kernel: {SVM_KERNEL}, C: {SVM_C}")
    # --- MODIFIED NOTE ---
    # Ensure features_final_for_clustering is available before proceeding
    # This print statement confirms which feature set is intended for use
    if features_final_for_clustering is not None:
         print(f"NOTE: SVM uses LOW-DIMENSIONAL features (shape: {features_final_for_clustering.shape}) used for clustering.")
    else:
         print("WARNING: Low-dimensional features (features_final_for_clustering) are None, SVM step might fail.")

    final_labels = [None] * total_images
    processed_clusters = 0

    # Check if necessary data is available before starting the loop
    if cluster_labels is None:
        print("Error: Cannot train SVMs as cluster_labels are not available.")
    elif features_final_for_clustering is None: # Explicit check for the feature set needed
         print("Error: Cannot train SVMs as low-dimensional features (features_final_for_clustering) are not available.")
    else:
        # Proceed with the loop using the imported Classifier class
        for cluster_id, group_indices in tqdm(clusters.items(), desc="Processing Clusters", unit="cluster"):
            processed_clusters += 1
            cluster_id = int(cluster_id) # Ensure integer ID

            # Handle Noise cluster
            if cluster_id == -1:
                for idx in group_indices: final_labels[idx] = 'Noise'
                continue

            # --- PROCESS ACTUAL CLUSTERS (Non-Noise) ---
            try:
                # Find the global indices of items within this cluster that have original labels
                tagged_indices_in_cluster_global = [idx for idx in group_indices if has_label[idx]]

                # --- Get LOW-DIMENSIONAL features for ALL items in this cluster ---
                # These features will be used for predicting labels for the whole cluster
                try:
                    # Select rows from features_final_for_clustering corresponding to group_indices
                    cluster_lowdim_features_arr = features_final_for_clustering[group_indices, :]
                except IndexError as idx_e:
                    # Error handling if indices in group_indices are out of bounds
                    print(f"\nE: IndexError preparing low-dim features for Cluster {cluster_id}: {idx_e}")
                    print(f"Max index in group_indices: {max(group_indices) if group_indices else 'N/A'}, Feature array shape: {features_final_for_clustering.shape}")
                    for idx in group_indices: final_labels[idx] = f'Error_FeatPrep_C{cluster_id}'
                    continue # Skip to the next cluster
                except Exception as feat_err:
                     # Catch other potential errors during feature preparation
                     print(f"\nE: Could not prepare low-dim features for Cluster {cluster_id}: {feat_err}")
                     for idx in group_indices: final_labels[idx] = f'Error_FeatPrep_C{cluster_id}'
                     continue # Skip to the next cluster

                # Check if there are any labeled items in this cluster
                if tagged_indices_in_cluster_global:
                    # --- Get LOW-DIMENSIONAL features and true labels ONLY for the tagged items ---
                    # These features will be used for training the SVM
                    try:
                        # Select rows from features_final_for_clustering for tagged items
                        tagged_lowdim_features_arr = features_final_for_clustering[tagged_indices_in_cluster_global, :]
                        # Get the corresponding true labels
                        true_labels = [actual_labels[idx] for idx in tagged_indices_in_cluster_global]
                    except IndexError as idx_e:
                         # Error handling if indices in tagged_indices are out of bounds
                         print(f"\nE: IndexError preparing tagged low-dim features for Cluster {cluster_id}: {idx_e}")
                         print(f"Max index in tagged_indices: {max(tagged_indices_in_cluster_global) if tagged_indices_in_cluster_global else 'N/A'}, Feature array shape: {features_final_for_clustering.shape}")
                         for idx in group_indices: final_labels[idx] = f'Error_TaggedFeatPrep_C{cluster_id}'
                         continue # Skip to the next cluster
                    except Exception as feat_err:
                         # Catch other potential errors preparing tagged features/labels
                         print(f"\nE: Could not prepare tagged low-dim features/labels for Cluster {cluster_id}: {feat_err}")
                         for idx in group_indices: final_labels[idx] = f'Error_TaggedFeatPrep_C{cluster_id}'
                         continue # Skip to the next cluster

                    # Determine the set of unique labels present in the tagged items
                    unique_true_labels = set(true_labels)

                    if len(unique_true_labels) == 1:
                        # CASE 1: All tagged items in the cluster have the same label.
                        # Assign this single label to all items (tagged and untagged) in the cluster.
                        pred_label = list(unique_true_labels)[0]
                        for idx in group_indices: final_labels[idx] = pred_label
                    elif len(unique_true_labels) > 1:
                        # CASE 2: Multiple unique labels exist among tagged items in the cluster.
                        # Train an SVM classifier.
                        try:
                            # Use the imported Classifier class
                            classifier = Classifier(kernel=SVM_KERNEL, C=SVM_C)
                            # --- Train on LOW-DIM tagged features ---
                            classifier.fit(tagged_lowdim_features_arr, true_labels)
                            # Predict labels for ALL items in the cluster using their LOW-DIMENSIONAL features
                            # --- Predict on LOW-DIM cluster features ---
                            predicted_labels = classifier.predict(cluster_lowdim_features_arr)
                            # Assign the predicted labels to the final_labels list
                            for i, idx in enumerate(group_indices):
                                final_labels[idx] = predicted_labels[i]
                        except Exception as fit_predict_err:
                             print(f"\nE: Error during SVM fit/predict for Cluster {cluster_id}: {fit_predict_err}")
                             import traceback
                             traceback.print_exc() # Print full traceback for debugging
                             for idx in group_indices: final_labels[idx] = f'Error_SVM_FitPred_C{cluster_id}'
                             continue # Skip to next cluster
                    else:
                        # CASE 3: Should not happen if tagged_indices_in_cluster_global is not empty,
                        # but handle defensively. It means tagged items existed but had no valid labels?
                         print(f"\nW: Cluster {cluster_id}: Tagged items found, but no unique labels? Assigning error label.")
                         for idx in group_indices: final_labels[idx] = f'Error_NoLabels_C{cluster_id}'

                else:
                    # CASE 4: No tagged (labeled) items were found in this cluster.
                    # Mark all items in this cluster as 'Untagged'.
                    for idx in group_indices: final_labels[idx] = f'Untagged_Cluster_{cluster_id}'

            except Exception as svm_e:
                # Catch any other unexpected errors during the processing of this cluster
                print(f"\nE: Unhandled error processing SVM logic for Cluster {cluster_id}: {svm_e}")
                import traceback
                traceback.print_exc() # Print detailed traceback for debugging
                # Assign a generic SVM error label to all items in this cluster
                for idx in group_indices: final_labels[idx] = f'Error_SVM_Generic_C{cluster_id}'
                continue # Continue to the next cluster
            # --- END OF Cluster Processing Try-Except ---
        # --- END OF Cluster Loop ---

    print(f"\nFinished processing {processed_clusters} clusters/noise groups.")
    print("-" * 25)


    # --- Note on Loss Visualization ---
    # (Note remains the same)
    print("\n--- Note on Loss Visualization ---")
    print("Standard DBSCAN+SVM pipeline doesn't have iterative training loss curves.")
    print("-" * 25)

    # --- Save Results ---
    print(f"\nSaving final labels to '{OUTPUT_FILE}'...")
    # (Saving and summary code remains the same)
    try:
        np.save(OUTPUT_FILE, np.array(final_labels, dtype=object), allow_pickle=True)
        print(f"Final labels saved successfully to {OUTPUT_FILE}")

        print("\n--- Final Label Summary ---")
        final_label_counts = defaultdict(int); none_count = 0; error_svm_count = 0; untagged_count = 0; noise_count = 0; assigned_label_count = 0; total_final = len(final_labels)
        for lbl in final_labels:
            if lbl is None: none_count += 1
            else:
                lbl_str = str(lbl); final_label_counts[lbl_str] += 1
                if lbl_str.startswith('Error_'): error_svm_count += 1
                elif lbl_str.startswith('Untagged_'): untagged_count += 1
                elif lbl_str == 'Noise': noise_count += 1
                elif not (lbl_str.startswith('Error_') or lbl_str.startswith('Untagged_') or lbl_str == 'Noise'): assigned_label_count +=1

        print(f"Total items processed: {total_final}")
        if none_count > 0: print(f"  - Items still None: {none_count}")
        print(f"  - Items labeled 'Noise': {noise_count} ({noise_count/total_final:.2%})")
        print(f"  - Items in Untagged Clusters: {untagged_count} ({untagged_count/total_final:.2%})")
        print(f"  - Items with SVM/Processing Errors: {error_svm_count} ({error_svm_count/total_final:.2%})")
        print(f"  - Items assigned a final label: {assigned_label_count} ({assigned_label_count/total_final:.2%})")

        display_labels = {k: v for k, v in final_label_counts.items() if not (k.startswith('Error_') or k.startswith('Untagged_') or k == 'Noise' or k == 'None')}
        sorted_labels = sorted(display_labels.items(), key=lambda item: item[1], reverse=True)
        print("\nAssigned Label Distribution (Top 20):")
        if not sorted_labels: print("  - No items were assigned actual final labels.")
        else:
            for lbl, count in sorted_labels[:20]: print(f"  - Label '{lbl}': {count} ({count/total_final:.2%})")
            if len(sorted_labels) > 20: print("  - ... (other assigned labels)")
    except Exception as e: print(f"Error saving final labels or generating summary: {e}")


# --- Execution Guard ---
if __name__ == "__main__":
    if IMAGE_DIR and not os.path.exists(IMAGE_DIR):
        print(f"Image directory '{IMAGE_DIR}' not found. Creating it.")
        try: os.makedirs(IMAGE_DIR); print(f"Created directory: {IMAGE_DIR}")
        except OSError as e: print(f"Warning: Could not create directory {IMAGE_DIR}: {e}")
    main()