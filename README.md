## KoSimpleEval

Simple evaluation kit for Korean and English benchmarks.

## How to use

Install required packages.

```python
pip install -r requirements.txt
```

Run the provided scripts

```
bash eval.sh
bash score.sh
```

To change the models of interest, open the eval.sh file and make the appropriate changes in ```models```.
For example, to evaluate llama-3-8b and qwen2.5-7b, make the following change.

```
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)
```


To change the benchmarks of interest, open the eval.sh file and make the appropriate changes in ```datasets```.  
(However, please be noted that the dataset should be added to [HAERAE-HUB/KoSimpleEval](https://huggingface.co/datasets/HAERAE-HUB/KoSimpleEval) first.)  
For example, to evaluate on KMMLU-Redux and Pro, make the following change.

```
datasets=(
  "KMMLU_Redux"
  "KMMLU_Pro"
)
```

## Adding new datasets. 

Leave an issue here. I'll add it for you.

## Citations

To cite this project you may use:

```
@article{son2025pushing,
  title={Pushing on Multilingual Reasoning Models with Language-Mixed Chain-of-Thought},
  author={Son, Guijin and Yang, Donghun and Patel, Hitesh Laxmichand and Agarwal, Amit and Ko, Hyunwoo and Lim, Chanuk and Panda, Srikant and Kim, Minhyuk and Drolia, Nikunj and Choi, Dasol and others},
  journal={arXiv preprint arXiv:2510.04230},
  year={2025}
}
```
