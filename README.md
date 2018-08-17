# Off-the-shelf Scikit-learn Gradient Boosting Machine

Wraps sklearn Gradient Boosting Regressor to:

1. automate modeling similar to gbm library in R

2. handle categorical features

3. overlay data and descriptive statistics in data visualization of partial dependencies for better inference


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need Scikit-learn library (http://scikit-learn.org/stable/install.html). Install it using your package manager, for example
```
conda install scikit-learn
```
or
```
pip install -U scikit-learn
```

### Installing

cd to the repo root directory and run:
```
pip install -e .
```

Try running an example:

```
import sklearn_gbm_ots.sklearn_gbm_wrapper_example as example
example.run_gbm()
```
you should observe log of building a model using cross-validation for tuning number of ensembles (trees) and resulting plots (displayed and saved to gbm_output sub-folder)

## Running the tests

Basic test only is developed so far.

```
python tests/test_sklearn_gbm_wrapper_example.py
```

### Break down into end to end tests

TBD

```
TBD
```

### And coding style tests

TBD

```
TBD
```

## Deployment

TBD

## Built With

 

## Contributing

Please read [CONTRIBUTING.md](https://link) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Alex Radnaev** - *Initial work*

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* TBD
