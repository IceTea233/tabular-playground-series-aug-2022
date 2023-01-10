# For 2023 NYCU introduction to ML Final Project

This repository is for my final project of the course in college. It is to solve a problem on Kaggle - [Tabular Playground Series - Aug 2022](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview)

## Requirements

This repository use `pipenv` for dependency management. Make sure that `pipenv` has been install.

```setup
pipenv install
```

Then run the following command to activate the virtual enviroment shell

```shell
pipenv shell
```

Or you may also install the dependencies manually.

- `python` (`3.9` is recommended)
- `numpy`
- `pandas`
- `scikit-learn`

## Training

To train the model, run this command:

```train
python train.py
```

After that, a file named `model.pk` should be generated by default. It should store the trained model.

## Inference

To generate the prediction with a train, run this command:

```predict
python predict.py
```

## Pre-trained Models

You can download pretrained models here:

- https://drive.google.com/file/d/1gEAnvDM2U53suH8ZdcbaCO2qlxhjkte1/view?usp=share_link

## Results

The model achieves the following performance on :

- validation score (ROC): 0.6047459583156607
- Public Score on Kaggle: 0.58601
- Private Score on Kaggle: 0.59003

## References

- This repository has taken the idea from the references below
  - [Solution](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/349810) posted by nour hadrich
    - 3 vs 2 cross validation
    - Logistic Regression as one possible model
- Data are from the [contest page on Kaggle](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data)