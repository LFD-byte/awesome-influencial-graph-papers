# awesome-influencial-graph-papers

A collection of research papers on graph.

## GCN

这里整理了一些图卷积文章，参考了 [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers) 和 [dgl](https://docs.dgl.ai/api/python/nn-pytorch.html#global-pooling-layers) 中的图卷积函数。
- Semi-Supervised Classification with Graph Convolutional Networks `GCNConv` `ICLR 2017` [\[pdf\]](https://arxiv.org/pdf/1609.02907.pdf) [\[code\]](https://github.com/tkipf/pygcn)
- Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering `ChebConv` `NIPS 2016` [\[pdf\]](https://arxiv.org/pdf/1606.09375.pdf) [\[code\]](https://github.com/mdeff/cnn_graph)
- Inductive Representation Learning on Large Graphs `SAGEConv` `NIPS 2017` [\[pdf\]](https://arxiv.org/pdf/1706.02216.pdf) [\[code\]](https://github.com/williamleif/GraphSAGE)
- Inductive Representation Learning on Large Graphs `CuGraphSAGEConv` `NIPS 2017` [\[pdf\]](https://arxiv.org/pdf/1706.02216.pdf) [\[code\]](https://github.com/williamleif/GraphSAGE)
- Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks `GraphConv` `AAAI 2019` [\[pdf\]](https://arxiv.org/pdf/1810.02244.pdf) [\[code\]](https://github.com/chrsmrrs/k-gnn)
- Learning Representations of Irregular Particle-detector Geometry with Distance-weighted Graph Networks `GravNetConv` `EPJ C 2019` [\[pdf\]](https://arxiv.org/pdf/1902.07987.pdf) [\[code\]](https://github.com/jkiesele/caloGraphNN)
- Gated Graph Sequence Neural Networks `GatedGraphConv` `ICLR 2016` [\[pdf\]](https://arxiv.org/pdf/1511.05493.pdf) [\[code\]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ggnn)
- Residual Gated Graph ConvNets `ResGateGraphConv` `ICLR 2018` [\[pdf\]](https://arxiv.org/pdf/1711.07553.pdf) [\[code\]](https://github.com/xbresson/spatial_graph_convnets)
- Graph Attention Networks `GATConv` `ICLR 2018` [\[pdf\]](https://arxiv.org/pdf/1710.10903.pdf) [\[code\]](https://github.com/PetarV-/GAT)
- Graph Attention Networks `CuGraphGATConv` `ICLR 2018` [\[pdf\]](https://arxiv.org/pdf/1710.10903.pdf) [\[code\]](https://github.com/PetarV-/GAT)
- Understanding GNN Computational Graph: A Coordinated Computation, IO, and Memory Perspective `FusedGATConv` `MLSys 2022` [\[pdf\]](https://arxiv.org/abs/2110.09524)
- How Powerful are Graph Neural Networks? `GATv2Conv` `ICLR 2019` [\[pdf\]](https://arxiv.org/pdf/1810.00826.pdf) [\[code\]](https://github.com/weihua916/powerful-gnns)
- Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification `TransformerConv` `IJCAI 2021` [\[pdf\]](https://arxiv.org/pdf/2009.03509.pdf) [\[code\]](https://github.com/PaddlePaddle/PGL/tree/main/ogb_examples/nodeproppred/unimp)
- Attention-based Graph Neural Network for Semi-Supervised Learning `AGNNConv` `ICLR 2018` [\[pdf\]](https://arxiv.org/pdf/1803.03735.pdf) [\[code\]](https://github.com/dawnranger/pytorch-AGNN)
- Topology Adaptive Graph Convolutional Networks `TAGConv` `arxiv 2017` [\[pdf\]](https://arxiv.org/pdf/1710.10370.pdf) [\[code\]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/tagcn)
- How Powerful are Graph Neural Networks? `GINConv` `ICLR 2019` [\[pdf\]](https://arxiv.org/pdf/1810.00826.pdf) [\[code\]](https://github.com/weihua916/powerful-gnns)
- Strategies for Pre-training Graph Neural Networks `GINEConv` `ICLR 2020` [\[pdf\]](https://arxiv.org/pdf/1905.12265.pdf) [\[code\]](https://github.com/snap-stanford/pretrain-gnns/)
- Graph Neural Networks with Convolutional ARMA Filters `ARMAConv` `IEEE Trans 2022` [\[pdf\]](https://arxiv.org/pdf/1901.01343.pdf) [\[code\]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/arma)
- Simplifying Graph Convolutional Networks `SGConv` `ICML 2019` [\[pdf\]](https://arxiv.org/pdf/1902.07153.pdf) [\[code\]](https://github.com/Tiiiger/SGC)
- Simple Spectral Graph Convolution `SSGConv` `ICLR 2021` [\[pdf\]](https://openreview.net/pdf?id=CYO5T-YjWZV) [\[code\]](https://github.com/allenhaozhu/SSGC)
- Predict then Propagate: Graph Neural Networks meet Personalized PageRank `APPNPConv` `ICLR 2019` [\[pdf\]](https://arxiv.org/pdf/1810.05997.pdf) [\[code\]](https://github.com/gasteigerjo/ppnp)
- Convolutional Networks on Graphs for Learning Molecular Fingerprints `MFConv` `NIPS 2015` [\[pdf\]](https://arxiv.org/pdf/1509.09292.pdf) [\[code\]](https://github.com/HIPS/neural-fingerprint)
- Modeling Relational Data with Graph Convolutional Networks `RGCNConv` `ESWC 2018` [\[pdf\]](https://arxiv.org/pdf/1703.06103.pdf) [\[code\]](https://github.com/tkipf/relational-gcn)
- Modeling Relational Data with Graph Convolutional Networks `FastRGCNConv` `ESWC 2018` [\[pdf\]](https://arxiv.org/pdf/1703.06103.pdf) [\[code\]](https://github.com/tkipf/relational-gcn)
- Modeling Relational Data with Graph Convolutional Networks `CuGraphRGCNConv` `ESWC 2018` [\[pdf\]](https://arxiv.org/pdf/1703.06103.pdf) [\[code\]](https://github.com/tkipf/relational-gcn)
- Relational Graph Attention Networks `RGAConv` `arxiv 2019` [\[pdf\]](https://arxiv.org/abs/1904.05811) [\[code\]](https://github.com/Cartus/AGGCN)
- Signed Graph Convolutional Network `SignedConv` `ICDM 2018` [\[pdf\]](https://arxiv.org/pdf/1808.06354.pdf) [\[code\]](https://github.com/benedekrozemberczki/SGCN)
- Just Jump: Towards Dynamic Neighborhood Aggregation in Graph Neural Networks `DNAConv` `arxiv 2019` [\[pdf\]](https://arxiv.org/abs/1904.04849) [\[code\]]
- PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation `PointConv` `CVPR 2017` [\[pdf\]](https://arxiv.org/pdf/1612.00593.pdf) [\[code\]](https://github.com/charlesq34/pointnet)
- Dynamic Graph CNN for Learning on Point Clouds `EdgeConv` `ACM Trans 2019` [\[pdf\]](https://arxiv.org/pdf/1801.07829.pdf) [\[code\]](https://github.com/WangYueFt/dgcnn)
- PointCNN: Convolution On X-Transformed Points `XConv` `NIPS 2018` [\[pdf\]](https://arxiv.org/pdf/1801.07791.pdf) [\[code\]](https://github.com/yangyanli/PointCNN)
- PPFNet: Global Context Aware Local Features for Robust 3D Point Matching `PPFConv` `CVPR 2018` [\[pdf\]](https://arxiv.org/pdf/1802.02669.pdf) [\[code\]](https://github.com/vinits5/learning3d)
- FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis `FeaStConv` `CVPR 2018` [\[pdf\]](https://arxiv.org/abs/1706.05206) [\[code\]](https://github.com/nitika-verma/FeaStNet)
- Point Transformer `PointTransformerConv` `ICCV 2021` [\[pdf\]](https://arxiv.org/abs/2012.09164) [\[code\]](https://github.com/yangyanli/PointCNN)
- Hypergraph Convolution and Hypergraph Attention `HyperGraphConv` `Pattern Recognit 2021` [\[pdf\]](https://arxiv.org/abs/1901.08150) [\[code\]](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/hypergraph_conv.py)
- ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations `LEConv` `AAAI 2020` [\[pdf\]](https://arxiv.org/abs/1911.07979) [\[code\]](https://github.com/malllabiisc/ASAP)
- Principal Neighbourhood Aggregation for Graph Nets `PNAConv` `NeurIPS 2020` [\[pdf\]](https://arxiv.org/pdf/2004.05718.pdf) [\[code\]](https://github.com/lukecavabarrett/pna)
- Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks `ClusterConv` `KDD 2019` [\[pdf\]](https://arxiv.org/pdf/1905.07953.pdf) [\[code\]](https://github.com/google-research/google-research)
- DeeperGCN: All You Need to Train Deeper GCNs `GENConv` `arxiv 2020` [\[pdf\]](https://arxiv.org/pdf/2006.07739.pdf) [\[code\]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/deepergcn)
- Simple and Deep Graph Convolutional Networks `GCN2Conv` `ICML 2020` [\[pdf\]](https://arxiv.org/pdf/2007.02133.pdf) [\[code\]](https://github.com/chennnM/GCNII)
- Path Integral Based Convolution and Pooling for Graph Neural Networks `PANConv` `NIPS 2020` [\[pdf\]](https://arxiv.org/pdf/2006.16811.pdf) [\[code\]](https://github.com/YuGuangWang/PAN)
- A Reduction of a Graph to a Canonical Form and an Algebra Arising During this Reduction `WLConv` `Nauchno-Technicheskaya Informatsiya 1968` [\[pdf\]](https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf) [\[code\]]
- GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation `FiLMConv` `ICML 2020` [\[pdf\]](https://arxiv.org/abs/1906.12192) [\[code\]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/GNN-FiLM)
- How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision `SurperGATConv` `ICLR 2021` [\[pdf\]](https://openreview.net/pdf?id=Wi5KUNlqWty) [\[code\]](https://github.com/dongkwan-kim/SuperGAT)
- Beyond Low-Frequency Information in Graph Convolutional Networks `FAConv` `AAAI 2021` [\[pdf\]](https://arxiv.org/abs/2101.00797) [\[code\]](https://github.com/bdy9527/FAGCN)
- Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions `EGConv` `arXiv 2021` [\[pdf\]](https://arxiv.org/abs/2104.01481) [\[code\]]
- Pathfinder Discovery Networks for Neural Message Passing `PDNConv` `WWW 2021` [\[pdf\]](https://arxiv.org/abs/2010.12878) [\[code\]](https://github.com/benedekrozemberczki/PDN)
- Design Space for Graph Neural Networks `GeneralConv` `NIPS 2020` [\[pdf\]](https://arxiv.org/pdf/2011.08843.pdf) [\[code\]](https://github.com/snap-stanford/graphgym)
- Heterogeneous Graph Transformer `HGTConv` `WWW 2020` [\[pdf\]](https://arxiv.org/pdf/2003.01332.pdf) [\[code\]](https://github.com/acbull/pyHGT)
- Heterogeneous Edge-Enhanced Graph Attention Network For Multi-Agent Trajectory Prediction `HEATConv` `arxiv 2021` [\[pdf\]](https://arxiv.org/abs/2106.07161) [\[code\]](https://github.com/Xiaoyu006/MATP-with-HEAT)
- Heterogenous Graph Attention Network `HANConv` `WWW 2019` [\[pdf\]](https://arxiv.org/pdf/1903.07293.pdf) [\[code\]](https://github.com/Jhy1993/HAN)
- LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation `LGConv` `SIGIR 2020` [\[pdf\]](https://arxiv.org/pdf/2002.02126.pdf) [\[code\]](https://github.com/gusye1234/pytorch-light-gcn)
- Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud `PointGNNConv` `CVPR 2020` [\[pdf\]](https://arxiv.org/pdf/2003.01251.pdf) [\[code\]](https://github.com/WeijingShi/Point-GNN)
- Recipe for a General, Powerful, Scalable Graph Transformer `GPSConv` `NIPS 2022` [\[pdf\]](https://arxiv.org/abs/2205.12454) [\[code\]](https://github.com/rampasek/GraphGPS)
- Anti-Symmetric DGN: a stable architecture for Deep Graph Networks `AntiSymmericConv` `ICLR 2023` [\[pdf\]](https://arxiv.org/abs/2210.09789) [\[code\]](https://github.com/gravins/anti-symmetricdgn)
- Edge Directionality Improves Learning on Heterophilic Graphs `DirGNNConv` `LoG 2023` [\[pdf\]](https://arxiv.org/abs/2305.10498) [\[code\]](https://github.com/emalgorithm/directed-graph-neural-network)
- MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing `MixHopConv` `ICML 2019` [\[pdf\]](https://arxiv.org/pdf/1905.00067.pdf) [\[code\]](https://github.com/samihaija/mixhop)
- Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs `GMMConv` `CVPR 2017` [\[pdf\]](https://arxiv.org/pdf/1611.08402.pdf) [\[code\]](https://github.com/dmlc/dgl/tree/master/examples/mxnet/monet)
- Neural Message Passing for Quantum Chemistry `NNConv` `ICML 2017` [\[pdf\]](https://arxiv.org/pdf/1704.01212.pdf) [\[code\]](https://github.com/brain-research/mpnn)


## Graph Pooling
常用的简单的图池化方法有 SumPooling, AvgPooling 和 MaxPooling，这里列举一些其他的更加复杂的图池化方法，参考了 [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers) 和 [dgl](https://docs.dgl.ai/api/python/nn-pytorch.html#global-pooling-layers) 中的图池化函数。
- An End-to-End Deep Learning Architecture for Graph Classification `SortPooling` `AAAI 2018` [\[pdf\]](https://ojs.aaai.org/index.php/AAAI/article/view/11782) [\[code\]](https://github.com/muhanzhang/DGCNN)
- Graph U-Nets `TopKPooling` `ICML 2019` [\[pdf\]](https://arxiv.org/abs/1905.05178) [\[code\]](https://github.com/HongyangGao/Graph-U-Nets) [\[code\]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/sagpool)
- Self-Attention Graph Pooling `SAGPooling` `ICML 2019` [\[pdf\]](https://arxiv.org/abs/1904.08082) [\[code\]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/sagpool)
- Gated Graph Sequence Neural Networks `GlobalAttentionPooling` `ICLR 2016` [\[pdf\]](https://arxiv.org/abs/1511.05493) [\[code\]](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ggnn)
- Order Matters: Sequence to sequence for sets `Set2SetPooling` `ICLR 2016` [\[pdf\]](https://arxiv.org/abs/1511.06391) [\[code\]](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/aggr/set2set.py)
- Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks `SetTransformerPooling` `ICML 2019` [\[pdf\]](https://arxiv.org/abs/1810.00825) [\[code\]](https://github.com/juho-lee/set_transformer)
- Towards Graph Pooling by Edge Contraction, Edge Contraction Pooling for Graph Neural Networks  `EdgePooling` `ICML 2019` `arxiv 2019` [\[pdf\]](https://mediatum.ub.tum.de/doc/1521739/document.pdf)[\[pdf\]](https://arxiv.org/abs/1905.10990) [\[code\]](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/pool/edge_pool.py)
- ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations `ASAPooling` `AAAI 2020` [\[pdf\]](https://arxiv.org/abs/1911.07979) [\[code\]](https://github.com/malllabiisc/ASAP)
- Path Integral Based Convolution and Pooling for Graph Neural Networks `PANPooling` `NIPS 2020` [\[pdf\]](https://arxiv.org/abs/2006.16811) [\[code\]](https://github.com/YuGuangWang/PAN)
- Memory-Based Graph Networks `MemPooling` `ICLR 2020` [\[pdf\]](https://arxiv.org/abs/2002.09518) [\[code\]](https://github.com/amirkhas/GraphMemoryNet)

## pytorch-geometric

pytorch-geometric (pyg) 的官方示例：

1. [Introduction: Hands-on Graph Neural Networks](https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing)
2. [Node Classification with Graph Neural Networks](https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing)
3. [Graph Classification with Graph Neural Networks](https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing)
4. [Scaling Graph Neural Networks](https://colab.research.google.com/drive/1XAjcjRHrSR_ypCk_feIWFbcBKyT4Lirs?usp=sharing)
5. [Point Cloud Classification with Graph Neural Networks](https://colab.research.google.com/drive/1D45E5bUK3gQ40YpZo65ozs7hg5l-eo_U?usp=sharing)
6. [Explaining GNN Model Predictions using Captum](https://colab.research.google.com/drive/1fLJbFPz0yMCQg81DdCP5I8jXw9LoggKO?usp=sharing)
7. [Customizing Aggregations within Message Passing](https://colab.research.google.com/drive/1KKw-VUDQuHhMo7sCd7ZaRROza3leBjRR?usp=sharing)
8. [Node Classification Instrumented with Weights&Biases](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pyg/8_Node_Classification_(with_W&B).ipynb)
9. [Graph Classification Instrumented with Weights&Biases](https://colab.research.google.com/github/wandb/examples/blob/pyg/graph-classification/colabs/pyg/Graph_Classification_with_PyG_and_W%26B.ipynb)
10. [Link Prediction on MovieLens](https://colab.research.google.com/drive/1xpzn1Nvai1ygd_P5Yambc_oe4VBPK_ZT?usp=sharing)
11. [Link Regression on MovieLens](https://colab.research.google.com/drive/1N3LvAO0AXV4kBPbTMX866OwJM9YS6Ji2?usp=sharing)

pyg 的指南项目：

1. Introduction \[[YouTube](https://www.youtube.com/watch?v=JtDgmmQ60x8),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial1/Tutorial1.ipynb)\]
2. PyTorch basics \[[YouTube](https://www.youtube.com/watch?v=UHrhp2l_knU),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial2/Tutorial2.ipynb)\]
3. Graph Attention Networks (GATs) \[YouTube](https://www.youtube.com/watch?v=CwsPoa7z2c8),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial3/Tutorial3.ipynb)\]
4. Spectral Graph Convolutional Layers \[[YouTube](https://www.youtube.com/watch?v=Ghw-fp_2HFM),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial4/Tutorial4.ipynb)\]
5. Aggregation Functions in GNNs \[[YouTube](https://www.youtube.com/watch?v=tGXovxQ7hKU),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial5/Aggregation%20Tutorial.ipynb)\]
6. (Variational) Graph Autoencoders (GAE and VGAE) \[[YouTube](https://www.youtube.com/watch?v=qA6U4nIK62E),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial6/Tutorial6.ipynb)\]
7. Adversarially Regularized Graph Autoencoders (ARGA and ARGVA) \[[YouTube](https://www.youtube.com/watch?v=hZkLu2OaHD0),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial7/Tutorial7.ipynb)\]
8. Graph Generation \[[YouTube](https://www.youtube.com/watch?v=embpBq1gHAE)\]
9. Recurrent Graph Neural Networks \[[YouTube](https://www.youtube.com/watch?v=v7TQ2DUoaBY),  [Colab (Part 1)](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial9/Tutorial9.ipynb),  [Colab (Part 2)](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial9/RecGNN_tutorial.ipynb)\]
10. DeepWalk and Node2Vec \[[YouTube (Theory)](https://www.youtube.com/watch?v=QZQBnl1QbCQ),  [YouTube (Practice)](https://youtu.be/5YOcpI3dB7I),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial11/Tutorial11.ipynb)\]
11. Edge analysis \[[YouTube](https://www.youtube.com/watch?v=m1G7oS9hmwE),  [Colab (Link Prediction)](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial12/Tutorial12%20GAE%20for%20link%20prediction.ipynb),  [Colab (Label Prediction)](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial12/Tutorial12%20Node2Vec%20for%20label%20prediction.ipynb)\]
12. Data handling in PyG (Part 1) \[[YouTube](https://www.youtube.com/watch?v=Vz5bT8Xw6Dc),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial14/Tutorial14.ipynb)\]
13. Data handling in PyG (Part 2) \[[YouTube](https://www.youtube.com/watch?v=Q5T-JdyVCfs),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial15/Tutorial15.ipynb)\]
14. MetaPath2vec \[[YouTube](https://www.youtube.com/watch?v=GtPoGehuKYY),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial13/Tutorial13.ipynb)\]
15. Graph pooling (DiffPool) \[[YouTube](https://www.youtube.com/watch?v=Uqc3O3-oXxM),  [Colab](https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial16/Tutorial16.ipynb)\]

