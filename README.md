# Transformer-based Fallacious Argument Classification in Political Debates

In this repository you find the code which was used for the experiments in our paper [Fallacious Argument Classification in Political Debates](https://www.ijcai.org/proceedings/2022/575) (IJCAI-ECAI 2022).

## Project Overview

This repository contains the code and data for the paper "Fallacious Argument Classification in Political Debates" by Pierpaolo Goffredo, Shohreh Haddadan, Vorakit Vorakitphan, Elena Cabrio, and Serena Villata.

## Abstract

This project addresses the challenging task of classifying fallacious arguments in political debates. The main contributions are:

1. A novel annotated resource of 31 political debates from U.S. Presidential Campaigns, containing 1628 annotated fallacious arguments across six main categories.
2. A neural architecture based on transformers for fallacious argument classification, outperforming state-of-the-art results and standard baselines.

## Dataset: ElecDeb60To16-fallacy

- Source: 31 debates from U.S. presidential election campaigns (1960-2016)
- Total fallacious arguments annotated: 1628
- Main fallacy categories: Ad Hominem, Appeal to Authority, Appeal to Emotion, False Cause, Slippery Slope, Slogans
- Additional features: Argument components and relations

## Methodology

- Task: Multi-class classification of fallacious arguments
- Models tested: BERT, RoBERTa, Longformer, Transformer-XL
- Best performing model: Longformer with joint loss and argumentation features
- Features used: Political discourse speech context, fallacious argument snippet, argument component and relation labels

## Results

| Model |JointLoss| Dataset | Precision | Recall | Macro avg F1-Score |
|-------|---------|---------|-----------|--------|---------------------|
| BERT | ❌ | Fallacy Main Category |  0.62| 0.55| 0.55|
| RoBERta | ❌ | Fallacy Main Category |  0.58 | 0.56 | 0.53 |
| Transfomer-XL | ❌ | Fallacy Main Category |  0.61 | 0.45 | 0.47 |
| Transfomer-XL (+ speech) | ✅ | Fallacy Main Category |  0.61 | 0.51 | 0.53 |
| Longformer | ❌ | Fallacy Main Category |  0.64 | 0.60 | 0.57 |
| Longformer (+ speech)| ✅ | Fallacy Main Category |  0.66 | 0.61 | 0.61 |
| Longformer (+ speech, comp)| ✅ | Fallacy Main Category | 0.88 | 0.81 | 0.83 |
| Longformer ((+ speech, rel)| ✅ | Fallacy Main Category | 0.87 | 0.81 | 0.83 |
| Longformer ((+ speech,comp, rel)| ✅ | Fallacy Main Category | 0.84 | 0.85 | **0.84** |

The best model (Longformer with argumentation features) achieved an F1-score of **0.84** on the main fallacy categories.

## Repository Structure



## Usage
The code runs under Python 3.6 or higher. The required packages are listed in the requirements.txt, which can be directly installed from the file:

```
pip install -r /path/to/requirements.txt
```

Our code is based on the transformer library version 4.18.0. See https://github.com/huggingface/transformers for more details.

## Citation

If you use this work, please cite:

```
@inproceedings{goffredo2022fallacious,
  title={Fallacious Argument Classification in Political Debates},
  author={Goffredo, Pierpaolo and Haddadan, Shohreh and Vorakitphan, Vorakit and Cabrio, Elena and Villata, Serena},
  booktitle={Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence (IJCAI-22)},
  pages={4143--4149},
  year={2022}
}
```

## Contact

For any questions or issues, please open an issue in this repository or contact the authors.