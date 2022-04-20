Sharpened Dynamic Pseudo Labeling
=========================
This repo provides the code for reproducing the experiments 
in Sharpened Dynamic Pseudo Labeling for Complex Affects Analysis. 
Sharpened Dynamic Pseudo Labeling (SDFL) is a semi-supervised learning method that can be 
used for multi-label complex affects recognition, which could adaptively set thresholds during pseudo-labeling process.


### Dependencies
```angular2html
pip install sklearn transformers nltk torch datasets
```

###Data

We use SemEval-2018 Task 1 English dataset to evaluate our method. There is a built-in function 
to download the dataset automatically.

### Usage
To train SDPL,
```
python bertweet.py  \
--output_dir ./model/sdpl \
--do_train \
--do_eval \
--do_predict \
--sharpen \
--semi \
--num_train_epochs 30 \
--unlabel_num 5838 \
--threshold 0.95 \
--pseudo_start 15 \
--pos_certainty_threshold 0.01 \
--neg_certainty_threshold 0.01 \
```

To train UPS,

```
python bertweet.py  \
--output_dir ./model/ups \
--do_train \
--do_eval \
--do_predict \
--onlyups \
--semi \
--num_train_epochs 20 \
--unlabel_num 5838 \
--threshold 0.95 \
--pseudo_start 10 \
--pos_certainty_threshold 0.01 \
--neg_certainty_threshold 0.01 \
```

To train vanilla pseudo labeling,
```
python bertweet.py  \
--output_dir ./model/naivepl \
--do_train \
--do_eval \
--do_predict \
--onaivepl \
--semi \
--num_train_epochs 20 \
--unlabel_num 5838 \
--threshold 0.95 \
--pseudo_start 10 \
--pos_certainty_threshold 0.01 \
--neg_certainty_threshold 0.01 \
```

To train a fully-supervised learning model with limited data (for comparison),
```
python bertweet.py  \
--output_dir ./model/supervise \
--do_train \
--do_eval \
--do_predict \
--num_train_epochs 20 \
--unlabel_num 5838 \
--threshold 0.95 \
--pseudo_start 10 \
--pos_certainty_threshold 0.01 \
--neg_certainty_threshold 0.01 \
```
For other changeable hyper-parameters, please see the argument parser part in codes.