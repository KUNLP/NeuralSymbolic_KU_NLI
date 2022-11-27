# Natural Language Inference using Dependency Parsing
Code for HCLT 2021 paper: *[Natural Language Inference using Dependency Parsing](https://koreascience.kr/article/CFKO202130060562801.page?&lang=ko)*

## Dependencies
- python 3.7
- PyTorch 1.9.0
- tokenizers 0.10.3
- Transformers 4.6.1


All code only supports running on Linux.

# Model Structure

<img src='model.png' width='1000'>



## Data

Korean Language Understanding Evaluation-Natural Language Inference: *[KLUE-NLI](https://klue-benchmark.com/tasks/68/data/description)*

## Train & Test

```
python run_NLI.py
```

## Results on KLUE-NLI

| Model | Acc |
|---|--------- |
| NLI w/ DP | 90.78% |
