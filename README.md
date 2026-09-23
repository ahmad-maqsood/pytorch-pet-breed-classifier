# PyTorch Pet Breed Classifier

An end-to-end PyTorch project: transfer learning and model comparison on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), from training to a deployed web app.

**Live demo:** https://pytorch-pet-breed-classifier.streamlit.app/

Upload a photo of a cat or dog and the app predicts one of 37 breeds.

## Overview

The dataset has 37 breed categories (25 dog, 12 cat), roughly 200 images each, with large variation in pose, scale, and lighting. Several pretrained models were compared as feature extractors, the best one was fine-tuned, and the final model was deployed as a Streamlit app.

No separate validation set was used; the train/test split from `torchvision.datasets.OxfordIIITPet` (`trainval` / `test`) was used throughout, with test accuracy monitored during training. This means reported test numbers may be slightly optimistic, since some decisions (epochs, model choice) were informed by watching them.

## Model comparison (feature extraction, 10 epochs, frozen backbone)

| Model | Test accuracy | Params | CPU latency (single image)* |
|---|---|---|---|
| **ResNet-50** | **91.22%** | ~23.5M | ~120–145 ms |
| EfficientNet-B0 | 85.33% | ~4M | ~50 ms |
| MobileNetV3-Large | unstable, dropped | ~4M | ~30 ms |

*Measured on a CPU-only laptop (Intel i5, 8th gen), batch size 1. Not representative of server or mobile hardware.

MobileNetV3-Large's test loss fluctuated significantly during training and was excluded from further experiments rather than debugged further.

ResNet-50 was chosen despite being the largest and slowest of the three: the accuracy gap (5.9 points over EfficientNet-B0) was well outside run-to-run noise, and the latency difference (well under a second either way) was judged acceptable for a single-image API use case rather than real-time video.

## Fine-tuning

Starting from the frozen-backbone ResNet-50 (91.22% test accuracy), the final residual block (`layer4`) was unfrozen and trained with a two-learning-rate optimizer (a smaller rate for the pretrained backbone, a larger rate for the head), to avoid disrupting the pretrained weights.

| Setup | Test accuracy | Train accuracy | Train/test gap |
|---|---|---|---|
| Frozen backbone (baseline) | 91.22% | 99.48% | ~8.3 pts |
| Unfrozen `layer4`, no augmentation | 82.36% | 94.32% | ~11.9 pts |
| **Unfrozen `layer4` + augmentation (final model)** | **90.77%** | 91.47% | **~0.7 pts** |

The final model's test accuracy is essentially unchanged from the frozen baseline, but the train/test gap shrank from ~8 points to under 1, indicating substantially less overfitting and likely better generalization to genuinely new photos. Test accuracy peaked slightly higher (91.93%) at epoch 2 before settling at 90.77% by epoch 5, suggesting a small amount of overfitting was still creeping back in by the final epoch.

Data augmentation on the training set used `RandomResizedCrop`, `RandomHorizontalFlip`, and the same normalization as the pretrained weights. The test set transform was left unaugmented throughout.

## Evaluation

The final model was tested on the untouched Oxford-IIIT Pets test set and additionally on a handful of custom photos not from the dataset, to get a rough sense of real-world behavior beyond the benchmark. Per-class accuracy and a confusion matrix were not computed for this version of the project.

## Project structure

```
pytorch-pet-breed-classifier/
├── custom_images/
├── data/                    # downloaded dataset (gitignored)
├── going_modular/
│   ├── data_setup.py
│   ├── engine.py
│   └── utils.py
├── models/                  # trained weights (only the final deployed model is committed)
├── runs/                    # TensorBoard logs
├── 01_project_intro_and_scripts.ipynb   
├── 02_data_exploration_and_feature_extraction.ipynb
├── 03_fine_tuning.ipynb
├── 04_evaluation.ipynb
├── app.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Reproducing this project

```bash
git clone https://github.com/ahmad-maqsood/pytorch-pet-breed-classifier.git
cd pytorch-pet-breed-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks in order (01 → 04). The dataset downloads automatically via `torchvision.datasets.OxfordIIITPet`. Training was done on Google Colab (T4 GPU); all other work was done locally on a CPU-only laptop.

To view experiment curves:
```bash
tensorboard --logdir runs
```

To run the app locally:
```bash
streamlit run app.py
```

## Tools

PyTorch, torchvision, TensorBoard (experiment tracking), Streamlit (deployment).

## Author

Ahmad Maqsood
[GitHub](https://github.com/ahmad-maqsood) · [LinkedIn](https://linkedin.com/in/ahmad-maqsood-0b86b8314)
