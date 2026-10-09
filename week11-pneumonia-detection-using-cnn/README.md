# Pneumonia Detection from Chest X-rays: A First CNN in PyTorch

### Can a Small Convolutional Neural Network, Trained from Scratch, Spot Pneumonia in a Single Chest X-ray?

**Domain:** Healthcare / Medical Image Classification **Algorithm:** Convolutional Neural Network (CNN, built in PyTorch) **Data:** 26,684 labelled chest X-rays from the RSNA Pneumonia Detection Challenge (Kaggle)

> **Disclaimer:** This project is a learning exercise about how CNNs work. It is **not a medical tool** and must never be used to diagnose anyone. The network is deliberately small and trained from scratch, so the results are modest. They show *what a first CNN can do*, not the best that is possible.

---

## Problem Statement

Given **one chest X-ray**, can a neural network say whether the patient has **pneumonia** (yes or no)?

One PyTorch model is built and judged against one "do nothing" baseline:

1. **CNN**: four convolution layers that learn to detect visual patterns (edges, textures, hazy patches), followed by a single linear layer that turns those patterns into one score
2. **`always_negative` baseline**: predict "no pneumonia" for every patient, with no learning at all

**Why the baseline matters:** only about 22.5% of the patients have pneumonia, so a model that always says "no pneumonia" is already right about 77.5% of the time. A useful CNN has to beat that and also **find the sick patients**. That is why this project tracks **recall** and **precision** next to accuracy.

**What this project practises:**

| Idea | Where it appears |
|---|---|
| Images as arrays and tensors (height, width, channels) | Preparing the data |
| Convolution, ReLU, max pooling, global average pooling | The model |
| Dropout (a way to reduce overfitting) | The model |
| Binary classification with `BCEWithLogitsLoss` | Training |
| Class imbalance, recall and precision | EDA 2, Training, Evaluation |
| Train / validation / test split without leakage | Preparing the data |
| Early stopping and keeping the best weights | Training |
| Confusion matrix and looking at real predictions | Evaluation, Sample predictions |

---

## Dataset

- **Source:** the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) on Kaggle. The X-rays were labelled by radiologists
- **Download:** with the Kaggle API (about 3.7 GB zip file)
- **Size:** 26,684 labelled X-rays, one per patient

| File or folder | Content |
|---|---|
| `stage_2_train_labels.csv` | One row per box drawn around a lung opacity, plus one row for every patient without a box. Columns: `patientId`, `x`, `y`, `width`, `height`, `Target` |
| `stage_2_detailed_class_info.csv` | The same rows with a more detailed class name for each X-ray |
| `stage_2_train_images/` | The 26,684 labelled X-rays, one `.dcm` (DICOM) file per patient |
| `stage_2_test_images/` | Competition test images **without labels**, so they are not used here |

Because the competition's own test images have no labels, the labelled images are split into train, validation and test sets inside the notebook.

**What is `Target`?** `Target = 1` means radiologists marked at least one area of **lung opacity** (a hazy patch that suggests pneumonia). `Target = 0` means no such area was marked.

**What is DICOM?** It is the standard file format for medical images. It stores the pixels plus extra scan information. The `pydicom` library reads it, and `.pixel_array` returns the image as a NumPy array.

---

## Key EDA Findings

**One patient can have several rows.** The labels file has 30,227 rows but only **26,684 unique patients**. A patient with pneumonia gets one row per box. The notebook collapses the table to **one row per patient** before splitting, so the same patient can never land in two sets.

**The missing box values are normal.** `x`, `y`, `width` and `height` are empty in 20,672 rows. That is exactly the number of patients without pneumonia: they have no box to describe.

**The classes are imbalanced.** Of 26,684 patients, 20,672 have no pneumonia and **6,012 have pneumonia (22.5%)**, about 3.4 healthy for every sick patient. This is why the `always_negative` baseline already reaches 77.5% accuracy.

**"No pneumonia" does not mean "healthy".** The detailed class file splits the patients into three groups:

| Group | Patients |
|---|---|
| No Lung Opacity / Not Normal | 11,821 |
| Normal | 8,851 |
| Lung Opacity (pneumonia) | 6,012 |

More than half of the "no pneumonia" patients have *some other abnormality*. So the CNN has to separate pneumonia from other problems, which is harder than separating sick from healthy.

**The difference is subtle.** Negatives mostly show dark, clear lung fields. Positives typically show hazy white patches. But some negatives also look cloudy, so the classes cannot be separated by eye or by a simple brightness rule.

**Possible shortcuts.** Images differ a lot in framing, brightness and contrast. Some carry text markers (such as "PORTABLE" or "AP") and medical devices (tubes, wires). A CNN can learn such shortcuts instead of the lungs because they correlate with how sick a patient is. This is worth remembering when reading the results.

**Resizing costs detail.** The raw images are 1024 x 1024 pixels. They are shrunk to 150 x 150, which keeps only **2.1% of the pixels**. Large opacities should survive, but fine detail blurs. This is a deliberate trade-off between speed and detail.

**What the EDA meant for the modelling:**

| Finding | Decision |
|---|---|
| Only 22.5% positives | Compare against the 77.5% baseline and track recall and precision |
| Several rows per patient | Reduce to one row per patient before splitting |
| "No pneumonia" includes abnormal chests | Expect a hard problem and modest results |
| Subtle visual differences | Use a CNN, which learns local patterns from pixels |
| Large raw images | Shrink to 150 x 150 for speed |

---

## Data Preprocessing

### Split: Train, Validation, Test

Each patient goes into exactly one set.

| Split | Share | Patients | Used for |
|---|---|---|---|
| Train | 70% | 18,678 | Learning the weights |
| Validation | 15% | 4,003 | Deciding when to stop training |
| Test | 15% | 4,003 | The final score and sample predictions, never touched during training |

```python
train_df, temp_df = train_test_split(
    patients, test_size=0.30, stratify=patients['Target'], random_state=42)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['Target'], random_state=42)
```

Two details protect the result:

- **Stratified split:** every set keeps the same share of pneumonia patients (0.225 in train, validation and test)
- **No leakage:** the split is done on the one-row-per-patient table, and three overlap checks all print an empty `set()`, so no patient appears in two sets

### Images to Arrays

Each X-ray is read from its DICOM file and resized to 150 x 150. All images of a split are stacked into one array of shape **(N, 150, 150, 1)**:

- `N`: number of images
- `150, 150`: height and width in pixels
- `1`: one colour channel, because an X-ray is grayscale (a colour photo has 3)

Pixels stay as integers from 0 to 255 (`uint8`) to save memory. They are turned into numbers between 0 and 1 *inside* the model. Loading all images takes about 3 minutes.

### Tensors and Batches

PyTorch convolutions expect the channel axis **before** height and width, so `permute(0, 3, 1, 2)` changes (N, 150, 150, 1) into (N, 1, 150, 150). Labels become floats because the loss function compares them with the model's decimal output.

| Loader | Batch size | Shuffled? |
|---|---|---|
| Train | 32 | Yes, so every epoch sees a new order |
| Validation | 64 | No |
| Test | 64 | No, so predictions can be matched back to the right X-ray |

---

## The Model

### The Idea Behind a CNN

A **convolution layer** slides a small window (here 3 x 3 pixels) over the image. At each position it multiplies the pixels by a set of learned numbers called a **filter**. The filter acts as a pattern detector: it responds strongly wherever the image looks like the pattern it has learned. The same filter is reused at every position, so a pattern can be found wherever it appears.

Layers are stacked. Early layers pick up simple things (edges, brightness changes). Later layers combine them into larger patterns (textures, shapes). For X-rays, the hope is that deeper layers react to hazy patches in the lungs.

A fully connected layer on the raw 22,500 pixels would need over 2.8 million weights for just 128 outputs. This whole CNN has only **240,385 parameters**, because the small filters are shared across the image.

### Building Blocks

| Block | What it does |
|---|---|
| `Conv2d(in, out, 3, padding=1)` | `out` filters of size 3 x 3. `padding=1` keeps height and width unchanged |
| `ReLU` | Replaces negative numbers with 0, which lets the network learn curved (non-linear) rules |
| `MaxPool2d(2)` | Keeps the largest value in each 2 x 2 block, halving height and width |
| `AdaptiveAvgPool2d(1)` | Averages each feature map to a single number (global average pooling) |
| `Dropout(0.5)` | During training, randomly zeroes half of the values so the network cannot lean on any single one. Switched off for validation and testing |
| `Linear(128, 1)` | Turns 128 numbers into one score |

### The Network Code

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, x):
        x = x.float() / 255
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(1)
```

### What Happens Inside `forward` (batch of B images)

| Step | Shape | Meaning |
|---|---|---|
| Input | (B, 1, 150, 150) | B grayscale images |
| Scale by 255 | (B, 1, 150, 150) | Pixel values 0 to 255 become 0 to 1 |
| Conv 1 + ReLU, pool | (B, 32, 75, 75) | 32 filters, then half the size |
| Conv 2 + ReLU, pool | (B, 64, 37, 37) | 64 filters, then half the size (75 / 2 rounded down) |
| Conv 3 + ReLU, pool | (B, 128, 18, 18) | 128 filters, then half the size (37 / 2 rounded down) |
| Conv 4 + ReLU | (B, 128, 18, 18) | 128 filters, no pooling after this one |
| Global average pool | (B, 128, 1, 1) | Each feature map becomes its average |
| Flatten, dropout, linear | (B, 1) | One score per image |
| `.squeeze(1)` | (B,) | One number per image |

The output is a **logit**: a raw score where large positive means "pneumonia" and large negative means "no pneumonia". It is **not** a probability yet. `torch.sigmoid(logit)` turns it into a probability between 0 and 1, and a probability above **0.5** is read as "pneumonia".

### Parameters

| Layer | Parameters | Calculation |
|---|---|---|
| Conv 1 | 320 | 1 x 32 x 3 x 3 + 32 |
| Conv 2 | 18,496 | 32 x 64 x 3 x 3 + 64 |
| Conv 3 | 73,856 | 64 x 128 x 3 x 3 + 128 |
| Conv 4 | 147,584 | 128 x 128 x 3 x 3 + 128 |
| Linear | 129 | 128 x 1 + 1 |
| **Total** | **240,385** | Matches the count printed by the notebook |

---

## Training

### Settings

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` (binary cross-entropy that takes the raw logit) |
| Optimiser | Adam, learning rate 0.001 |
| Batch size | 32 (train) |
| Maximum epochs | 30 |
| Early stopping | Patience 5, judged on the validation loss |
| Dropout | 0.5 |
| Decision threshold | 0.5 |
| Random seed | Not set, so numbers change slightly from run to run |

**Why this loss?** For a yes/no label, the loss measures how surprised the model is by the right answer. A confident correct prediction gives a loss near 0, and a confident wrong one gives a large loss. `BCEWithLogitsLoss` applies the sigmoid internally, which is why the model outputs logits.

**What happens in every batch of 32 images:**

1. `model(xb)`: forward pass, get the logits
2. `loss_fn(logits, yb)`: compute the loss
3. `opt.zero_grad()`: clear the gradients from the last batch
4. `loss.backward()`: backpropagation, work out how each weight affects the loss
5. `opt.step()`: Adam nudges every weight slightly in the helpful direction

For the validation set only steps 1 and 2 run, and no weights change.

**How early stopping works here:** after each epoch (one full pass over the 18,678 training images) the validation loss is measured. If it improves, a copy of the weights is saved. If it fails to improve 5 epochs in a row, training stops. At the end, **the best saved weights are reloaded**, so the model that gets tested is the best one on validation, not simply the last one trained.

### Metrics Tracked

- **Recall** = TP / (TP + FN): of all patients who really have pneumonia, the share the model finds. If 100 are sick and the model finds 40, recall is 0.40
- **Precision** = TP / (TP + FP): of all patients the model flags, the share who are really sick. If it flags 100 and 65 are sick, precision is 0.65

Recall matters most here, because missing a sick patient is worse than a false alarm. The `always_negative` baseline has a recall of exactly 0, so any recall above 0 shows that the CNN learned something.

Training recall and precision are measured with dropout switched **on**, so they can look slightly worse than they would with dropout off. Validation numbers are measured with dropout off.

### Training Result

| | Epoch 1 | Epoch 30 |
|---|---|---|
| Training loss | 0.523 | 0.373 |
| Validation loss | 0.515 | 0.403 |
| Training recall / precision | 0.01 / 0.41 | 0.48 / 0.69 |
| Validation recall / precision | 0.10 / 0.42 | 0.48 / 0.62 |

- **Early stopping never triggered.** All 30 epochs ran, and the best validation loss (0.403) came at epoch 30. The model was still improving when it hit the epoch limit, so it is probably under-trained
- **No clear overfitting.** Training loss (0.373) and validation loss (0.403) stay close, though the gap widens slightly in the last epochs
- **Validation recall is jumpy.** In the last five epochs it was 0.36, 0.23, 0.36, 0.58 and 0.48. This is normal, because each point is measured on a slightly different version of the weights

---

## How the Model Is Evaluated

All final numbers use the **test set**: 4,003 patients never used for learning the weights or for early stopping.

The **confusion matrix** counts the four possible outcomes:

| | Predicted: no pneumonia | Predicted: pneumonia |
|---|---|---|
| **Truly no pneumonia** | True negative (TN): correct | False positive (FP): false alarm |
| **Truly pneumonia** | False negative (FN): a **missed** sick patient | True positive (TP): correct |

| Metric | What it tells you |
|---|---|
| **Accuracy** | Share of all patients classified correctly |
| **Recall** (sensitivity) | Share of sick patients found |
| **Precision** | Share of flagged patients who are really sick |
| **Specificity** | Share of healthy patients correctly cleared, TN / (TN + FP) |
| **ROC AUC** | The chance that a randomly chosen sick patient gets a higher probability than a randomly chosen healthy one. 0.5 is guessing, 1.0 is perfect ranking. It does not depend on the 0.5 threshold |

---

## Results

### Test Set (4,003 patients: 902 with pneumonia, 3,101 without)

| Model | Accuracy | Recall | Precision | Specificity | ROC AUC |
|---|---|---|---|---|---|
| `always_negative` baseline | 0.7747 | 0.00 | 0.0000 | 1.0000 | 0.5000 |
| **CNN** | **0.8129** | **0.48** | **0.6073** | **0.9097** | **0.8371** |

Working back from these numbers, the CNN's confusion matrix is roughly: about 2,821 true negatives, 280 false positives, 469 false negatives and 433 true positives.

### Sample Predictions

The notebook also shows 8 real test X-rays (4 sick, 4 healthy) with the true label, the predicted label and the model's probability. It is for building intuition only, because 8 images are far too few to measure accuracy. A healthy X-ray with p = 0.05 is a confident correct answer. A sick X-ray with p = 0.40 is a *near miss*: the model leaned towards pneumonia but did not cross the 0.5 line.

---

## Honest Interpretation

**The CNN learned something real, but it is not a screening tool.**

- **Accuracy: 81.3% vs 77.5%.** The gain looks small, but accuracy can only grow a few points above this baseline, so it is the wrong thing to judge by
- **Recall: 0.48 vs 0.** The CNN finds about 48 of every 100 pneumonia patients and misses about 52. That is far above the baseline, but still far too many misses for any real use
- **Precision: 0.61.** Roughly 3 out of every 5 flagged patients really have pneumonia
- **ROC AUC: 0.84 vs 0.5.** The probabilities do rank sick patients above healthy ones well. Because the AUC is much better than the recall at the 0.5 threshold suggests, a better threshold could improve recall without retraining (at the cost of more false alarms)
- **The typical mistake is a missed case.** On the test set there were about 469 missed pneumonia cases against about 280 false alarms, so the CNN is cautious about calling pneumonia

**Why the results are modest (and expected to be):**

1. **Small images.** Only about 2% of the original pixels are kept
2. **Trained from scratch.** With 18,678 images and no pretraining, the network has to learn everything from zero
3. **Mixed negatives.** Many "no pneumonia" X-rays show other abnormalities
4. **Under-trained.** The validation loss was still falling at epoch 30
5. **No class weighting or data augmentation**

**Lessons learned:**

1. **Shapes are everything.** Most of the work was tracking (N, H, W, C) versus (N, C, H, W) and how each layer changes the size
2. **Accuracy can lie.** With 22.5% positives, the baseline already scores 77.5%, so recall and precision told the real story
3. **Splitting by patient matters.** Several rows per patient would have leaked information between the sets
4. **Logits, sigmoid and thresholds are three separate things.** The model outputs a raw score, the sigmoid turns it into a probability, and the threshold turns that into a decision. Changing the threshold trades recall against precision without retraining
5. **Looking at real predictions helps.** A table says "recall 0.48", but the sample grid shows which X-rays are missed

---

## Notebook Structure

```
Introduction
└── Question, CNN vs always_negative baseline

Data
├── RSNA dataset, files, Target, DICOM
└── Kaggle download (Colab secrets)

Libraries

Exploratory Analysis
├── EDA 1: Label files (rows, missing values, duplicates)
├── EDA 2: One row per patient and class balance
├── EDA 3: What is inside the "no pneumonia" group
├── EDA 4: What the X-rays look like
├── EDA 5: One raw image and what resizing costs
└── What the EDA means for the modelling

Preparing the Data
├── Stratified train / validation / test split (no leakage checks)
└── Images to arrays (N, 150, 150, 1)

PyTorch Setup
└── Device selection, tensors, DataLoaders

The Model
├── The idea behind a CNN
├── Building blocks, shapes, parameter count
└── CNN class (4 conv layers + global average pooling + linear)

Training
├── Settings, loss, metrics
├── Training loop with early stopping
└── Learning curves

Evaluation on the Test Set
├── Confusion matrix
└── Accuracy, recall, precision, specificity, ROC AUC vs baseline

Sample Predictions
└── 8 real test X-rays with predictions

Verdict, Limitations and Next Steps
```

---

## How to Run

1. Create a free Kaggle account, open the [competition page](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) and **accept the rules once**
2. Create a Kaggle API token and save it in **Google Colab Secrets** under the name `KAGGLE_API_TOKEN`
3. Open `Pneumonia_Detection_using_CNN.ipynb` in Google Colab
4. **Switch to a GPU:** `Runtime → Change runtime type → GPU` (the saved run used the CPU, which works but is slow)
5. Run all cells in order (`Runtime → Run all`)

**Notes:**

- The download is about 3.7 GB and only needs to run once per session (it took about 30 seconds in the saved run)
- Loading and resizing all 26,684 images takes about 3 minutes
- No random seed is set, so your numbers will differ slightly from the ones above
- The notebook uses `google.colab`, so running it outside Colab needs a small change to how the Kaggle token is read

**Requirements:**

```
numpy
pandas
matplotlib
scikit-learn
torch
pydicom
opencv-python
tqdm
kaggle
```

Install with:

```
pip install numpy pandas matplotlib scikit-learn torch pydicom opencv-python tqdm kaggle
```

---

## What I Would Do Next

- **Train longer:** the validation loss was still falling at epoch 30. Raise the epoch limit and let early stopping decide
- **Use bigger images:** try 224 x 224 or larger so subtle opacities are not lost
- **Transfer learning:** start from a network already trained on millions of images (for example ResNet or DenseNet from `torchvision`)
- **Data augmentation:** small random rotations, brightness and contrast changes, and zoom
- **Fix the threshold and the loss:** choose the threshold on the *validation* set using a precision-recall curve, or give positives more weight with `BCEWithLogitsLoss(pos_weight=...)`
- **Check for shortcuts:** use Grad-CAM heat maps to see whether the CNN looks at the lungs or at text markers and tubes
- **Use the bounding boxes:** this model only answers *whether* there is pneumonia. Object detection (for example Faster R-CNN or YOLO) can also say *where*
- **Repeat with several seeds:** fix `torch.manual_seed` and train several times to see how much of the result is luck

---

*Part of the [AI-ML-Journey](https://github.com/Himanshu13-Programs/AI-ML-Journey) repository.*
