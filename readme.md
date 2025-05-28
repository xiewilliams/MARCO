# MARCO: A Cooperative Knowledge Transfer Framework for Personalized Cross-domain Recommendations

Recommender systems (RecSys) frequently encounter data sparsity issues, particularly when addressing cold-start scenarios involving new users or items. Multi-source cross-domain recommendation (CDR) addresses these challenges by transferring valuable knowledge from multiple source domains to enhance recommendations in a target domain. However, existing reinforcement learning (RL)-based CDR methods typically rely on a single-agent framework, leading to negative transfer issues caused by inconsistent domain contributions and inherent distributional discrepancies among sources. To overcome these limitations, MARCO, a novel Multi-Agent Reinforcement Learning-based Cross-Domain recommendation framework, is proposed. It leverages cooperative multiagent reinforcement learning (MARL), where each agent is dedicated to estimating the contribution from an individual source domain, effectively managing credit assignment and mitigating negative transfer. In addition, an entropy-based action diversity penalty is introduced to enhance policy expressiveness and stabilize training by encouraging diverse agents’ joint actions. Extensive experiments across four benchmark datasets demonstrate MARCO’s superior performance over state-of-the-art methods, highlighting its robustness and strong generalization capabilities. 


## Introduction
This repository provides the implementations of MARCO.


## Requirements

- Python 3.8
- Pytorch 
- tensorflow 
- Pandas
- Numpy
- Tqdm


## Code Structure
```bash
MARCO   
├── README.md                                 Read me file
├── rawdata                                   Raw data set
├── preprocessing                             Parsing the raw data and segmenting the mid data
│  ├── preprocess.py                          Demo on how to process the Amazon multiple data sets
├── multidomain_data                          Final data set
├── singleMF                                  Pretraining on single domain dataset 
│  ├── config.json                            Config for MF 
│  ├──singlepre.py                            Single domain MF model                       
│── pretrainmodel                             Pretrained_pt_files
├── config.json                               Config for MARCO model
├── entry.py                                  Training entry script
├── run.py                                    Run functions
├── models.py                                 Meta-learning models for cross-domain recommendation
├── rl_environment.py                         Reinforcement learning(RL) environment
├── MAPPO.py                                  Multi-agent RL Models 
├── rl_trainer_infer.py                       RL trainer and inference module
```

## Installation
```shell
pip install -r requirements.txt

```

## Dataset

We utilized the Amazon Reviews 5-score dataset. 
To download the Amazon dataset, you can use the following link: [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html).  
Download the three domains: [CDs and Vinyl](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz), [Movies and TV](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz), [Books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz), [Electronics](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz)
(5-scores), and then put the data in `./rawdata`.

You can use the following command to preprocess the dataset. 
The two-phase data preprocessing includes parsing the raw data and segmenting the mid data. 
The final data will be under `./multidomain_data`.


```python
python preprocess.py
```



# Run

Pretrained model on single-domain data.
```python
python singlepre.py
```



These  arguments are available for users to adjust MARCO:

```bash
--target_domain, 0:Book|1:CD|2:MV|3:Electronic
--task_run, targetcd|targetmv|targetbook|targetel
--seed, input random seed
--ratio, split ratio for given task
--gpu, gpu id
--epoch, epoch number
--lr, learning rate
--use_cuda, whether to user gpu
--rl_lr, learning rate for reinforcement learning
```

You can run this model through:

```powershell
# Run directly with default parameters 
python entry.py

# Reset training epoch to `10`
python entry.py --epoch 20

# Reset several parameters
python entry.py --gpu 1 --lr 0.02

# Reset seed (we use seed in[900, 1000, 10, 2020, 500])
python entry.py --seed 900
```