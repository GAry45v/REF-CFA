# REF-CFA: Anomaly-Related Residual Fields for Cross-domain Anomaly Detection

[](https://pytorch.org/)
[](https://opensource.org/licenses/MIT)

**REF-CFA** is a novel framework for label-free anomaly detection under domain shift. It addresses the challenge where conventional residuals are dominated by stochastic noise. By explicitly modeling the **Residual-Evolution Field (REF)**, this method isolates the persistent, anomaly-aligned signal from the non-anomalous dynamics. A **Cross-domain Field Alignment (CFA)** module then aligns these field representations across domains, enabling a detector trained on a labeled **Source Domain** to be robustly reused on an unlabeled **Target Domain**.

## ✨ Key Components

  * **Residual-Evolution Field (REF)**:
      * Constructs a spatio-temporal vector field from diffusion residuals using three primitives:
          * **Gradient Residual ($R_t$)**: Captures amplitude discrepancies.
          * **Directional Offset ($M_t$)**: Measures deviations orthogonal to the normal manifold flow.
          * **Path-Integrated Drift ($Q_t$)**: Accumulates persistent drift over the reverse trajectory.
      * **Field Transformer**: A lightweight sequence model that aggregates these primitives to produce **Adaptive Statistics** (Energy $E$, Non-Stationarity $NS$, Directional Variability $DV$).
  * **Cross-domain Field Alignment (CFA)**:
      * Enables domain reuse via **Temporal Alignment**, **Second-order Feature Alignment**, and **Directional Subspace Alignment**.
  * **Supervised Field Detector**:
      * A discriminative head ($f_\psi$) trained on the Source Domain using the extracted REF features.

## 🛠️ Installation

It is recommended to use Anaconda to create a virtual environment:

```bash
conda create -n refcfa python=3.8
conda activate refcfa

# Install PyTorch (Adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy opencv-python pyyaml matplotlib scipy scikit-learn tqdm omegaconf
```

## 📂 Data Preparation

Please ensure your data directory structure matches the following layouts and update the `root` path in your config files.

### 1\. MVTec AD

Standard MVTec structure:

```text
/data/mvtec/
├── bottle
│   ├── train
│   │   └── good
│   └── test
│       ├── good
│       ├── broken_large
│       └── ...
```

### 2\. VisA (Visual Anomaly)

Structure required for VisA:

```text
/data/visa/
├── candle
│   ├── Data
│   │   ├── Images
│   │   │   ├── Normal
│   │   │   └── Anomaly
│   │   └── Masks
│   │       └── Anomaly
```

### 3\. DAGM 2007

DAGM structure with `_def` suffix for defective samples:

```text
/data/dagm/
├── Class1
│   ├── 0001.png
│   └── ... (Normal samples)
├── Class1_def
│   ├── 0001.png
│   └── 0001_mask.png (Defective samples and masks)
```

## 🚀 Getting Started

### 1\. Stage S1 & T1: Score Network Training

Train the diffusion score networks ($S_{\theta_S}, S_{\theta_T}$) for both Source and Target domains.

```bash
# Train Source Score Network (e.g., MVTec Bottle)
python tools/train_net.py --config-file configs/experiments/mvtec/source_bottle.yaml

# Train Target Score Network (e.g., MVTec Cable)
# Note: Target training is unsupervised (on unlabeled mixture)
python tools/train_net.py --config-file configs/experiments/mvtec/target_cable.yaml
```

### 2\. Stage S2: Source REF & Detector Training

Construct the **Residual-Evolution Field (REF)** on the source domain and train the **Supervised Field Detector** ($g_\phi, f_\psi$).

```bash
# Train the Field Transformer and Detector on Source
python tools/analyze_net.py \
  --config configs/experiments/mvtec/source_bottle.yaml \
  --checkpoint output/MVTec_Source_Bottle_Train/checkpoint_epoch_2000.pth \
  --train_localization
```

  * This step learns to isolate anomaly-aligned components ($R, M, Q$) and optimizes the supervised loss $\mathcal{L}_S$.

### 3\. Stage T2: Target CFA & Inference

Apply **Cross-domain Field Alignment (CFA)** to align the target REF to the source space, then reuse the detector for inference.

```bash
# Run Inference on Target Domain using Source Detector + CFA
python tools/analyze_net.py \
  --config configs/experiments/mvtec/target_cable.yaml \
  --checkpoint output/MVTec_Source_Bottle_Train/checkpoint_epoch_2000.pth \
  --run_supervised_localization
```

  * This applies Temporal, Second-order, and Directional alignment before prediction.

### 4\. Visualization & Analysis

#### REF Visualization (Residual Dynamics)

Visualize the evolution of the residual primitives ($R_t, M_t, Q_t$) and the stationarity break in anomalous regions.

```bash
python tools/analyze_net.py \
  --config configs/experiments/visa/target_pcb2.yaml \
  --checkpoint output/VisA_Source_Candle_Train/checkpoint_epoch_2000.pth \
  --analyze_residuals
```

  * Results saved in `output/vis_ref_dynamics/`.

#### Field Transformer Analysis

Evaluate the **Adaptive Statistics** ($E, NS, DV$) generated by the Field Transformer.

```bash
python tools/analyze_net.py \
  --config configs/experiments/mvtec/target_cable.yaml \
  --checkpoint output/MVTec_Source_Bottle_Train/checkpoint_epoch_2000.pth \
  --analyze_transformer
```

## ⚙️ Configuration

Configurations are located in `configs/` and use an inheritance mechanism.

  * `base/runtime.yaml`: Global training parameters.
  * `base/models.yaml`: Model architecture (UNet, Diffusion Steps).
  * `experiments/<dataset>/<config>.yaml`: Experiment-specific configurations.

## 📊 Output Structure

| Component | Description | Output Path |
| :--- | :--- | :--- |
| **Score Network** | $S_\theta$ weights | `output/<ExpName>/checkpoint_epoch_X.pth` |
| **REF Detector** | Field Transformer & Head weights | `output/.../supervised_segmenter.pth` |
| **Inference** | Anomaly Maps $A(u)$ | `output/vis_supervised/` |
| **Analysis** | REF Dynamics ($R, M, Q$ plots) | `output/vis_ref_dynamics/` |

## 📝 Citation

If you use this code, please cite our paper:

```bibtex
@article{refcfa2026,
  title={Anomaly-Related Residual Fields for Cross-domain Anomaly Detection},
  author={Anonymous},
  journal={CVPR Submission},
  year={2026}
}
```