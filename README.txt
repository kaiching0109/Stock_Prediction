@Author: Kai Ching Suen
@Data: 4/20/2019

START:

# In project root
$ pip freeze > requirements.txt
# And to install the packages
$ pip install -r requirements.txt

FILES:

├── README.md          <- Front page of the project. Let everyone
│                         know the major points.
│
├── models             <- Trained and serialized models, model
│                         predictions, or model summaries.
│
├── notebooks          <- Jupyter notebooks. Use set naming
│                         E.g. `1.2-rd-data-exploration`.
│
├── reports            <- HTML, PDF, and LaTeX.
│   └── figures        <- Generated figures.
│   └── docs
│
├── requirements.txt   <- File for reproducing the environment
│                         `$ pip freeze > requirements.txt`
├── data
│   ├── processed      <- The final data sets for modelling.
│   └── raw            <- The original, immutable data.
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module.
    │
    ├── utility <- General functions to import.
    |    └── custom_func.py
    │
    ├── features       <- Scripts raw data into features for
    │   │                 modeling.
    │   └── feature_builder.py
    │
    ├── models         <- Scripts to train models and then use
    │   │                 trained models to make predictions.
    │   │
    │   ├── prediction_controller.py
    │   └── train_controller.py
    │
    └── visualizations            <- Scripts to create visualizations.
        └── vizualizer.py

REFERENCE:
1. project file structures:
  https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e
2.
