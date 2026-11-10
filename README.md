# Custom SEM Micro-graph ML Adjustment Software

A desktop editor for reviewing scanning electron microscope (SEM) micrographs. Open a TIFF, examine surface detail at different magnifications, and tune its grayscale rendering. The editor can learn from examples you approve and suggest a complete set of adjustments for the next image. Image processing and training run locally.

## Install and launch

From the directory containing `sem_editor.py`, create and activate a Python virtual environment. On macOS or Linux, run the following commands.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment in PowerShell.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the four runtime dependencies and start the dashboard.

```bash
pip install PyQt6 opencv-python numpy scikit-learn
python sem_editor.py
```

Launch from the same working directory each session to use the same training log.

## First image to first prediction

1. **Open a TIFF.** Select **Open TIFF** and choose a single-page, 8-bit-per-channel grayscale, RGB or RGBA micrograph. The editor converts 24-bit RGB pixels to one 8-bit grayscale intensity per pixel. Grayscale pixels retain their original values. For 16-bit acquisitions or stacks, first export an 8-bit single-frame working copy from the acquisition software.
2. **Inspect a feature.** Scroll over the viewport to zoom. Drag the image to pan, or select **Fit** to see the entire frame. Compare fine edges and even-textured areas while tuning the sliders in the collapsible left panel. While a slider is being dragged, large frames use a draft preview. Release the slider and wait for **Full resolution** to inspect fine structures. **Reset** restores the neutral rendering.
3. **Approve the rendering.** Once satisfied, select **Log Good Image** in the ML Control Hub. Each approval adds a row to `sem_ml_dataset.csv` in the working directory, then commits and pushes the CSV to the current branch on GitHub in the background. Logging settings and exporting a processed image are separate actions.
4. **Build a training set.** Repeat with several source images. Include examples with different brightness, noise and local contrast representative of the specimens you expect to process. Five approved rows unlock prediction.
5. **Run Auto-Adjust.** Open another micrograph and select **Auto-Adjust**. The editor learns from the current log and moves the sliders to its suggested settings. Inspect the preview, refine individual settings and log the result if you approve it.
6. **Export a copy.** Choose **Save TIFF As** to export the full-resolution, single-channel image as a separate TIFF. The export contains processed pixels and omits acquisition metadata. Keep the acquisition TIFF for intensity measurements and its instrument metadata.

## Processing controls

Each render starts from the loaded grayscale matrix and applies the enabled filters in the order below.

| Stage | Controls and interpretation |
| --- | --- |
| Median filter | **Median radius (px)** selects a square neighborhood of width `2 × radius + 1`. A radius of 1 replaces each pixel with the median of a 3 × 3 patch, suppressing isolated spikes. Zero bypasses the filter. |
| Anisotropic diffusion | **Diffusion iterations** sets the number of smoothing passes (0–20). **Diffusion threshold** sets the intensity-gradient scale for conduction. A higher threshold allows more smoothing across edges. |
| Black and white points | **Black point** and **White point** map the selected intensity interval to the full grayscale range. Intensities outside the interval saturate. Narrowing the interval raises contrast and can hide distinctions at either end. |
| Global gamma | **Gamma × 100** sets the nonlinear lookup table applied to the clipped levels. At 100, midtones retain their values. Higher values brighten midtones and lower values darken them. |
| CLAHE | **CLAHE clip × 10** sets the local histogram clip limit (20 means 2.0, zero bypasses). **Tiles per axis** sets the equalization grid. More tiles make contrast more local and may amplify noise. |
| Unsharp mask | **Unsharp amount (%)** scales the high-frequency residual (zero bypasses). **Unsharp sigma × 10 (px)** sets the Gaussian scale used to compute that residual (10 means 1 px). Inspect bright edges for halos. |

## What Auto-Adjust learns

The feature extractor describes the **unadjusted grayscale image** with 273 numbers. The first 256 are normalized histogram bins, one per possible 8-bit intensity. Next comes the variance of the Laplacian, a measure of high-frequency response sensitive to both edges and noise. The last 16 are pixel-intensity variances in a 4 × 4 spatial grid, ordered left to right and top to bottom.

Columns `hist_000` through `hist_255`, `laplacian_variance`, and `variance_r0_c0` through `variance_r3_c3` hold the source descriptors. Columns beginning with `slider_` hold integer control coordinates. Each CSV row pairs that vector with the approved slider coordinates. Auto-Adjust fits a 96-tree multi-output random forest that predicts all coordinates together. Targets are scaled by their slider ranges during fitting and mapped back to valid controls for display. The five-row threshold permits a first prediction. Its accuracy depends on how well the logged images and operator choices represent the new field of view.

If Auto-Adjust reports too few examples, approve more images in the current working directory. If previously recorded examples seem missing, check which directory you launched from.

## Training data on GitHub

**Log Good Image** saves each example locally before attempting a Git commit and push. The ML hub reports the result. If the network, authentication or a Git conflict prevents a push, your examples remain on disk and training continues to use them. Click **Sync Training Data** to retry without logging a duplicate example. The next approval also retries any pending local commits.

Launch from your cloned project directory, with Git installed and authenticated for the `origin` remote. Set your Git author name and email before the first commit. Automatic commits include only `sem_ml_dataset.csv`. The lock file remains ignored. With a public repository, the CSV's image-derived measurements and approved slider values are publicly accessible. Image files remain ignored.

If a push reports that the remote branch has newer commits, reconcile those changes using Git and click **Sync Training Data** again. The editor uses a normal push and preserves local history on failure.

## Publish the repository

With Git installed, a Git author identity configured and the GitHub CLI authenticated, run the standalone deployment script from its download directory in a bash or zsh terminal.

```bash
bash deploy.sh
```

To choose the parent directory, pass its path as the first argument.

```bash
bash deploy.sh "$HOME/Documents"
```

The script creates the project folder `SEM Processing Software`, writes the application and README, initializes Git, commits the files and pushes to a public repository named `SEM-Processing-Software` under the authenticated account.

## Research background

- Yan and colleagues frame photo adjustment as learning an edit from examples and use image descriptors that capture local context. This editor adapts that learning objective to operator-approved SEM slider values using a random forest. [Automatic Photo Adjustment Using Deep Neural Networks](https://arxiv.org/abs/1412.7725v2)
- Li and colleagues study differing behavior in under- and over-exposed image regions using learned color features. Their work gives context for examining regional variation in addition to a global histogram. Here the regional descriptors measure grayscale intensity variance. [Color Shift Estimation-and-Correction for Image Enhancement](https://arxiv.org/abs/2405.17725v1)
- Van der Walt and colleagues describe scientific image processing with Python arrays. This editor uses NumPy for grayscale matrices and OpenCV for image operations. [scikit-image: image processing in Python](https://doi.org/10.7717/peerj.453)


##Instructions to use
Start by adjusting one image manually. Open TIFF → adjust the sliders → Log Good Image teaches the software what you like. Save TIFF As saves the image you see. You can do either or both.
1. Open a micrograph
If the editor is closed, launch it from Terminal:
cd "$HOME/Documents/SEM Processing Software" && .venv/bin/python sem_editor.py
Click Open TIFF and choose a single-page, 8-bit grayscale or 24-bit RGB TIFF. The editor displays RGB images in grayscale. If your microscope produces 16-bit TIFFs or image stacks, export an 8-bit, single-frame working copy first and keep the original acquisition file.
2. Adjust the image
Use the mouse wheel to zoom in on a feature you recognize, such as a particle edge. Click and drag to pan. For your first image, start with Gamma and leave the other sliders at their defaults until you need them.
If the image looks…	Try this
Too dark or too bright	Under Tone, move Gamma × 100 above 100 to brighten midtones or below 100 to darken them.
Washed out	Raise Black point slightly or lower White point. Watch the darkest and brightest areas as you move the sliders. Stop while you can still distinguish detail in both.
Flat in just part of the image	Under Local contrast, try CLAHE clip × 10 at 20. That represents a clip limit of 2.0. Check whether it brings out texture or mainly amplifies noise.
Speckled with isolated bright or dark pixels	Under Denoise, try Median radius at 1. Inspect your smallest features to see whether they survive the filter.
Grainy across otherwise smooth areas	Try a few Diffusion iterations. Start with the default Diffusion threshold and inspect a nearby edge as you increase the iterations.
Slightly soft at the edges	Under Sharpen, raise Unsharp amount (%) gradually. Back it down if bright outlines appear around particles.


While you drag a slider, large images can show a draft preview. Release the slider and wait for “Full resolution” before judging fine detail. Fit shows the whole image. Reset returns every slider to its starting value.
3. Save the image or teach the model
- Save TIFF As exports a full-resolution grayscale TIFF. Choose a new name such as sample_01_adjusted.tif. Keep the acquisition TIFF for measurements and microscope metadata.
- Log Good Image records the original image’s brightness and texture measurements alongside your current slider values. Click it once when you’re satisfied. If you launched from the folder above, the examples are stored in Documents/SEM Processing Software/sem_ml_dataset.csv.
4. Try Auto-Adjust
Manually adjust and log five different representative micrographs. Then open a sixth image and click Auto-Adjust in the right sidebar. The model moves all ten sliders to suggested settings. Zoom in, inspect the result, and fine-tune it if needed. Click Log Good Image if you want the final settings to become another training example, and Save TIFF As if you want an exported copy.

Five logged examples unlock the button’s prediction workflow. To improve its suggestions, log a variety of images that resemble the specimens and acquisition conditions you actually work with.
