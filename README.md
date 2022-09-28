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

Table 1: Results on text classification tasks.
<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Accuracy</th>
    <th rowspan="2">Parameters (M)</th>
  </tr>
  <tr>
    <th>AGNews</th>
    <th>Amazon</th>
    <th>DBpedia</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>XLNet</td>
    <td>95.55</td>
    <td>/</td>
    <td>99.40</td>
    <td>240</td>
  </tr>
  <tr>
    <td>UDA</td>
    <td>/</td>
    <td>96.50</td>
    <td>98.91</td>
    <td>/</td>
  </tr>
  <tr>
    <td>BERT Large</td>
    <td>/</td>
    <td>97.37</td>
    <td>99.36</td>
    <td>340</td>
  </tr>
  <tr>
    <td>BERT-ITPT-FiT</td>
    <td>95.20</td>
    <td>/</td>
    <td>99.32</td>
    <td>/</td>
  </tr>
  <tr>
    <td>pNLP-Mixer XS</td>
    <td>89.62</td>
    <td>90.38</td>
    <td>98.24</td>
    <td>0.404</td>
  </tr>
  <tr>
    <td>pNLP-Mixer XL</td>
    <td>90.45</td>
    <td>90.56</td>
    <td>98.40</td>
    <td>6.0</td>
  </tr>
  <tr>
    <td>HBA-Mixer-2</td>
    <td>91.30</td>
    <td>93.28</td>
    <td>93.49</td>
    <td>0.13</td>
  </tr>

  </tr>
</tbody>
</table>

Table 2: Results on semantic analysis tasks.
<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Accuracy</th>
    <th rowspan="2">Parameters (M)</th>
  </tr>
  <tr>
    <th>Hyperpartisan</th>
    <th>IMDb</th>
    <th>Yelp-2</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>RoBERTa</td>
    <td>87.40</td>
    <td>95.30</td>
    <td>/</td>
    <td>125</td>
  </tr>
  <tr>
    <td>Longformer</td>
    <td>94.80</td>
    <td>96.70</td>
    <td>/</td>
    <td>149</td>
  </tr>
  <tr>
    <td>XLNet</td>
    <td>/</td>
    <td>96.21</td>
    <td>98.63</td>
    <td>240</td>
  </tr>
  <tr>
    <td>BERT Large</td>
    <td>/</td>
    <td>95.49</td>
    <td>/</td>
    <td>340</td>
  </tr>
  <tr>
    <td>UDA</td>
    <td>/</td>
    <td>95.80</td>
    <td>97.95</td>
    <td>/</td>
  </tr>
  <tr>
    <td>pNLP-Mixer XS</td>
    <td>89.80</td>
    <td>81.90</td>
    <td>84.05</td>
    <td>2.2/1.2/0.403</td>
  </tr>
  <tr>
    <td>pNLP-Mixer XL</td>
    <td>89.20</td>
    <td>82.90</td>
    <td>91.70</td>
    <td>8.4/6.8/4.9</td>
  </tr>
  <tr>
    <td>HBA-Mixer-2</td>
    <td>77.86</td>
    <td>86.79</td>
    <td>92.81</td>
    <td>8.5/2.2/0.12</td>
  </tr>
</tbody>
</table>
  
Table 3: Results on natural language inference.


Table 1: Main results of MHBA-Mixer on several datasets with hidden dimension 256.
  
| DATASET |MAX SEQ LEN | ACCURACY (%) | PARAMETERS (M) |
|  ----  | ---- |--------------|  ----  |
| AGNEWS | 128 | 91.79        | 0.726 |
| AMAZON |128| 91.88        | 0.726|
| DBPEDIA |128 | 98.44        | 0.726|
| HYPERPARTISAN |2048 | 89.43        | 0.698|
| IMDB |1024 | 87.88        | 0.677 |
| Yelp2 | 128| 92.57        | 0.726 |
| SST2 | 128| 83.48        | 0.726 |
| CoLA | 128| 69.51        | 0.726 |
| QQP | 128 | 82.02        | 0.726 |
  
Table 2: Main results of MHBA-Mixer with different hidden dimension.  
  
<table>
<thead>
  <tr>
    <th rowspan="2">Hidden Dimension</th>
    <th colspan="2">AGNews</th>
    <th colspan="2">IMDb</th>
    <th colspan="2">SST-2</th>
  </tr>
  <tr>
    <th>Accuracy (%)</th>
    <th>Parameters (M)</th>
    <th>Accuracy (%)</th>
    <th>Parameters (M)</th>
    <th>Accuracy (%)</th>
    <th>Parameters (M)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>64</td>
    <td>91.30</td>
    <td>0.10</td>
    <td>87.08</td>
    <td>0.10</td>
    <td>83.21</td>
    <td>0.10</td>
  </tr>
  <tr>
    <td>128</td>
    <td>91.42</td>
    <td>0.25</td>
    <td>87.76</td>
    <td>0.24</td>
    <td>82.63</td>
    <td>0.25</td>
  </tr>
  <tr>
    <td>256</td>
    <td>91.79</td>
    <td>0.73</td>
    <td>87.88</td>
    <td>0.68</td>
    <td>83.48</td>
    <td>0.73</td>
  </tr>
</tbody>
</table>  

Table 3: Main results of HBA-Mixer-2 and MHBA-Mixers.  
