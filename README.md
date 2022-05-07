# DeepLearningRC
Reproducibility Challenge for Deep Learning

# Introduction
In this repository, we attempt to reproduce the NeurIPS 2020 paper [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496)

## Requirements
* Python 3
* torch 1.10.0
* transformers 4.18.0

## Experiments 
Run the script with different arguments for different tasks 

A quick example is provided here: https://colab.research.google.com/drive/1Z6ype5ci6y8_Qy7wLJ665X3fRjpGgs9L#scrollTo=0oVHkwRHSWyt


Firstly, to download the data required for the experiments: 
```bash
python3 download-all-data.py
```

To run the experiments on the GLUE datasets: 

Example:

```bash
python3 run_glue_experiments.py --model_size "base" --task_name "cola" --experiment_type "fine-tuning" 
```

To run the experiments on the SQuAD datasets: 

Example:

```bash
python run_squad.py --model_size "small" --task_name "squadv1"
```

# References

[1] Z. Jiang, W. Yu, D. Zhou, Y. Chen, J. Feng, and S. Yan, “ConvBERT: Improving BERT with Span-based Dynamic Convolution.” 2020. (https://arxiv.org/abs/2008.02496) \
[2] A. Vaswani et al., “Attention is all you need,” in Advances in Neural Information Processing Systems, 2017, pp. 5998--6008. (https://arxiv.org/abs/1706.03762)
