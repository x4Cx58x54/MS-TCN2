# Data protocols for MSTCN++

This document explains format of preprocessed dataset for MSTCN++.

```
data
├── gtea
│   ├── features
│   │   ├── video_1.npy
│   │   ├── video_2.npy
│   │   └── ...
│   ├── groundTruth
│   │   ├── video_1.txt
│   │   ├── video_2.txt
│   │   └── ...
│   ├── mapping.txt
│   └── splits
│       ├── test.split1.bundle
│       ├── test.split2.bundle
│       ├── train.split1.bundle
│       ├── train.split2.bundle
│       └── ...
└── ...
```

* `video_x.npy`: features for `video_x`, shape is (`n_features`, `n_frame`).
* `video_x.txt`: frame-wise temporal label for `video_x`, contains `n_frame` lines, each line is the name of the action, and all background action is `background`.
* `mapping.txt`: contains `n_action` lines, each line is an integer (action number) and the action name, separated by a space.
* `train.splitx.bundle`: contains `train_dataset_size` lines, each line is a label file name in train dataset.

## Examples

### `data/gtea/groundTruth/S1_Cheese_C1.txt`

```
background
background
background
background
background
background
background
background
background
background
background
background
take
take
take
take
take
...
```

### `data/gtea/mapping.txt`

```
0 take
1 open
2 pour
3 close
4 shake
5 scoop
6 stir
7 put
8 fold
9 spread
10 background
```

### `data/gtea/splits/test.split1.bundle`

```
S1_Cheese_C1.txt
S1_CofHoney_C1.txt
S1_Coffee_C1.txt
S1_Hotdog_C1.txt
S1_Pealate_C1.txt
S1_Peanut_C1.txt
S1_Tea_C1.txt
```
