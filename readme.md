# meseg

this is pytorch-**me**dical-**seg**mentation (meseg)  repository.



table of contents:

1. about dataset
2. tutorial
3. experimental result

<br>

## About Dataset
### datasets
- 


<br>

## Tutorial

### 1. how to train your model

1. Clone this repository and install the dependency.

   ```bash
   git clone https://github.com/Team-Ryu/pytorch-medical-classification.git
   cd pytorch-medical-classification
   pip3 install requirements.txt # pass now
   ```

2. Make a symbolic link for datasets and pre-trained weights.

   ```bash
   ln -s ~/shared/hdd_ext/nvme1/medical data
   ```

3. Check `argparser`'s 3 important default options in `train.py`: `project_name`, `who`, and `output_dir`.

4. Check `config/train.json` files. 

   - `data_dir`: locations for each dataset in the project folder.
   - `model_list`: whole trainable model name list. If you don't specify `model_names`, this list will be trained.
   - `model_weight`: pre-trained weight paths.
   - `settings`: experiment settings (e.g. lr, optimizer, weight-decay). Improve this.
   
5. Choose 1) settings and 2) models and 3) GPU devices and train your model following the example command formats.

   ```bash
   # Single GPU
   python3 train.py -s setting1 setting2 -m model1 model2 -c 0
   # Multi GPU
   torchrun --nproc_per_node=2 --master_port=12345 train.py \
   -s setting1 setting2 -m model1 model2 -c 0,1
   ```
   
   *Tips*
   
   1. You should use different learning rates for different batch sizes (see [linear lr scaling rule](https://arxiv.org/pdf/1706.02677.pdf), [square lr scaling rule](https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change)). 
   
   2. The `train.py` load your settings from `config/train.json`, however, you can overwrite it by passing different command line arguments, except `model_weight` and `data_dir`.
   
   3. Important experiment setting:
   
      1. `--use-wandb`: If you want to write logs in the WanDB, you should add this option.
      2. `--model-names name1 name2`: If you want to run the whole model, you just skip this one.
      3. `--valid-freq N (default: None)`: specify the period of validation every N iter.
      4. `--save-weight`, `--save-metric metric_name`: save best model in terms of `metric_name`.
      5. `--metric-names metric_name1 metrci_name2`: evaluate model in these metric terms.
   
   

### 2. how to validate your model

1. check `config/valid.json`. We summarize overall structure of valid configuration. You should add your validation settings.

   ```json
   {
     "data_dir": { 
       "isic2018": "data/classification/skin/ISIC-2018"
     },
     "isic2018_v1": {
       "settings": {"dataset_type": "isic2018", "test_size": [224, 224], "center_crop_ptr": 0.875},
       "model_weight": {
         "convnext_tiny": "log/isic2018_v1_convnext_tiny_v1/best_weight.pth",
         "convnext_small": "log/isic2018_v1_convnext_small_v1/best_weight.pth",
         "convnext_base": "log/isic2018_v1_convnext_base_v1/best_weight.pth"
       }
     }
   }
   ```
   
2. Choose 1) model weight version id, 2) model lists, 3) split between valid and test, and 4) GPU index, and validate your model.

   ```bash
   python3 valid.py -s isic2018_v1 isic2019_v1 -c 2 --mode valid --use-wandb
   python3 valid.py -s isic2018_v1 isic2019_v1 -c 2 --mode test 
   ```
   
   *Tips*
   
   1. `--use-wandb`: It will automatically save validation table in to wandb (recommend to use).
   2. `isic2018`, `isic2019` : if `mode` is set to `test`, inference results will be saved in `log_val/*/ensemble.csv`.
   3. `ddsm` : if `mode` is set to `test`, `test` split will be used.
   4. Except `isic2018`, `isic2019`, `ddsm`, no dataset support test split.
   


### 3. how to add a new dataset

1. Create your dataset class file in the `mecla/dataset` folder and add `@register_dataset` decorator.

2. Import your dataset class in the `mecla/dataset/__init__.py` file.

3. Update `choice` option for `argparser.dataset_type` in `train.py` and `data_dir` in the `config/*.json` file.



## Experiment Result

| No   | Name        | auc(%) | #train(K) | #val(K) | #test(K) | #class | multilabel | sensor | pathology |
| ---- | ----------- | ------ | --------- | ------- | -------- | ------ | ---------- | ------ | --------- |
| 1    | CheXpert    |        | 191.0     | 0.2     | 0.0      | 5      | 1          | X-ray  | Lung      |
| 2    | ChestX-Ray8 |        | 86.5      | 25.6    | 0.0      | 14     | 1          | X-ray  | Lung      |
| 3    | DDSM+CBIS   |        | 55.0      | 7.7     | 7.7      | 2      | 0          | CT     | Breast    |
| 4    | VinDr-Mammo |        | 16.4      | 4.1     | 0.0      | 5      | 0          | CT     | Breast    |
| 5    | ISIC2018    |        | 10.0      | 0.2     | 0.6      | 7      | 0          | RGB    | Skin      |
| 6    | ISIC2019    |        | 20.3      | 5.1     | 8.2      | 9      | 0          | RGB    | Skin      |
| 7    | Messidor-2  |        | 1.4       | 0.4     | 0.0      | 5      | 0          | RGB    | Eye       |
| 8    | EyePACS     |        | 28.1      | 7.0     | 0.0      | 5      | 0          | RGB    | Eye       |
| 9    | PCAM        |        | 262.1     | 32.8    | 0.0      | 2      | 0          | RGB    | Lymph     |



### CheXpert

single model results

| No   | model name          | param | imagenet | acc  |
| ---- | ------------------- | ----- | -------- | ---- |
| 1    | ResNet50            | 25.6  | 79.8     | 89.5 |
| 2    | ResNet101           | 44.6  | 81.3     | 89.6 |
| 3    | ResNet152           | 60.2  | 81.8     | 89.3 |
| 4    | Denesenet121        | 8.0   | 75.6     |      |
| 5    | Denesenet161        | 28.7  | 77.4     |      |
| 6    | Denesenet169        | 14.2  | 75.6     |      |
| 7    | Denesenet201        | 20.0  | 77.3     |      |
| 8    | Inception-ResNet-v2 | 55.8  | 80.5     |      |
| 9    | ConvNext-T          | 29.0  | 82.1     | 90.3 |
| 10   | ConvNext-S          | 50.0  | 83.1     |      |
| 11   | ConvNext-B          | 89.0  | 83.8     |      |
| 12   | DeiT-S              | 22.0  | 79.8     | 87.2 |
| 13   | DeiT-B              | 86.0  | 81.8     | 87.8 |
| 14   | Swin-T              | 28.3  | 81.3     | 88.3 |
| 15   | Swin-S              | 49.6  | 83.0     | 88.8 |
| 16   | Swin-B              | 87.7  | 83.5     |      |

ensemble of models results

| no   | model name                | # model | acc  |
| ---- | ------------------------- | ------- | ---- |
| 1    | ResNet                    | 3       | 90.6 |
| 2    | ConvNext                  | 3       |      |
| 3    | DeiT                      | 2       | 88.6 |
| 4    | Swin                      | *2*     | 89.0 |
| 5    | ResNet+ConvNext           | *4*     | 91.0 |
| 6    | DeiT+Swin                 | 5       |      |
| 7    | ResNet+DeiT               | 5       | 90.5 |
| 8    | ConvNext+Swin             | 6       |      |
| 9    | ResNet+ConvNeXt+DeiT+Swin | 11      |      |
