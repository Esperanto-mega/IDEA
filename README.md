Paradigm Shift of GNN Explainer from Label Space to Prototypical Representation Space. ICLR 2026, Poster.
<div align="center">
  <img width="678" height="574" alt="image" src="https://github.com/user-attachments/assets/86bad68a-ca6e-47fc-a5c5-dba729d7bf8b" />
</div>

- --

### Environment
The environment has been exported to `IDEA/requirements.txt`.

### Dataset
Regrading the four real-wolrd datasets, i.e., Mutag, Alkane, Fluoride, Benzene, the original resource is available at [Harvard Dataverse](https://doi.org/10.7910/DVN/KULOS8), which is released by [_Evaluating explainability for graph neural networks_](https://www.nature.com/articles/s41597-023-01974-x).

Furthermore, the processed datasets alongside the ground explanantions are uploaled in `IDEA/data/`.

### Model
- The architecture of the target GNN to be explained in defined in `IDEA/gnn_model.py`. In `IDEA/GNN`, we upload the weights of trained GNN models.
- The RQ-VAE module is implemented in `IDEA/RQ-VAE.py`. Afterwards, the hierarchical graph tokenizer (HGTokenizer) is implemented in `IDEA/graph_tokenizer.py`.
- The IDEA explainer with PGExplainer as backbone is implemented in `IDEA/explainer/idea.py`. In addition, we upload the weights of pre-trained codebooks, including both shallow and deep branches, in `IDEA/codebook`.

### Evaluation
Based on `IDEA/explainer/eval_idea.py`, one can optimize and evaluate the IDEA explainer. The launching command belike,
```python
python IDEA/explainer/eval_idea.py --dataset Benzene --root_path IDEA/ --epochs 10 --lr 0.0005 --device -1 --batch_size 64 --codebook_size 32 --gt_learning_rate 0.01 --gt_epoch 10
```

- --
```bibtex
@inproceedings{idea2026,
  title     = {{Paradigm Shift of GNN Explainer from Label Space to Prototypical Representation Space}},
  author    = {Yin, Jun and Wang, Senzhang and Luo, Ziluowen and Huo, Peng and Yan, Hao and Miao, Hao and Li, Chaozhuo and Pan, Shirui and Zhang, Chengqi},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=X7eYISNf01}
}
```
