# Decoding Motor Intent from sEMG — Hand Gesture Classification

Machine-learning decoders that classify six hand gestures from two-channel surface EMG, comparing a classical linear decoder with neural networks and testing how well each generalises across days.

*Dheemant Jallepalli — 42-675 Fundamentals of Computational Biomedical Engineering, Carnegie Mellon University*

## Why

After spinal cord injury or stroke, many people still produce weak but measurable muscle activity. Surface EMG records that activity non-invasively, and a good decoder can turn it into control commands for prosthetics, exoskeletons and human–computer interfaces.

## Data

[sEMG for Basic Hand Movements](https://archive.ics.uci.edu/) (UCI ML Repository, Sapsanis et al., 2013)

- 1 able-bodied participant, 3 sessions (days)
- 2 channels: flexor carpi ulnaris and extensor carpi radialis, 500 Hz
- 6 gestures × 100 repetitions per day, 5 s per trial (1,800 trials)

## Pipeline

1. **Pre-processing:** band-pass (15–500 Hz) and 50 Hz notch filtering (from the dataset authors), sliding windows with 50 % overlap (100 samples for LDA/MLP, 128 for the CNN), full-wave rectification and normalisation
2. **Decoders:** Linear Discriminant Analysis (LDA), Multi-Layer Perceptron (MLP) and a 1-D CNN
3. **Evaluation:** within-day (80/20 split per day) and cross-day (train on day 1, test on all days)

## Results

- The **MLP beat LDA by ~20 % within-day and ~15 % cross-day**
- About **60 % six-class accuracy** when the decoder is recalibrated each day
- Accuracy drops across days for every model, most likely from electrode shift, skin-impedance changes and having only two channels

## Repository contents

| File | What it is |
|---|---|
| `Classifier_V2_Formatted.ipynb` | Final, cleaned decoder pipeline |
| `CNN_MLP_LDA_Until_Project3_IML.ipynb` | Earlier experiments with CNN, MLP and LDA |
| `Documentation.pdf` | Short write-up: background, methods, results, discussion |
| `Comp_BME_Dheemant (1).pdf` | Presentation slides |

## Next steps

More channels, CNN / transfer-learning decoders for better cross-day robustness, and multi-degree-of-freedom control.
