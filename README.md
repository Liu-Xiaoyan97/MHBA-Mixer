# MHBA-Mixer
MHBA-Mixer: an Efficient Mixer for NLP

## Architecture of MHBA-Mixer
![Arcitecture of MHBA-Mixer](./figure/MHBA-Mixer.jpg)
## details for hidden bias attention (HBA)
![(left) hidden bias attention (HBA), (right) Multi-Head HBA with n heads](./figure/Multi-Head%20HBA.jpg)

## How to train
`python main.py -d=YOUR_DATASET -t=train -p=YOUR_MODEL`  
`YOUR_DATASET` must be selected in `configs/nlp/*.yml`  
when train your model, `-p` is optional.

## How to test
`python main.py -d=YOUR_DATASET -t=test -p=YOUR_MODEL`  
`-p` must be a specific model in  `trained-models/*.ckpt`  
We provide 9 datasets which have been displayed in Table 1. 
## Experiments
Table 1: Main results of MHBA-Mixer on several datasets with hidden dimension 256.
  
| DATASET |MAX SEQ LEN | ACCURACY (%) | PARAMETERS (M) |
|  ----  | ---- | ----  |  ----  |
| AGNEWS | 128 | 91.79 | 0.726 |
| AMAZON |128|? | 0.726|
| DBPEDIA |128 |98.44 | 0.726|
| HYPERPARTISAN |2048 | 89.43| 0.698|
| IMDB |1024 |87.88 | 0.677 |
| Yelp2 | 128| 92.57 | 0.726 |
| SST2 | 128| 83.48 | 0.726 |
| CoLA | 128| 69.51 | 0.726 |
| QQP | 128 | 82.02 | 0.726 |
