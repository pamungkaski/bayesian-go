# Bayesian Go
Bayesian implementation using Golang

## Naive Bayes

In machine learning, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

### How to run

make sure to properly install golang and go path


```
$ go run app/main.go
```

prediction is int the `TebakanTugas1ML.csv`

### Alogrithm

First the CSV for Train data are loaded. Each item inputed to Naive Bayes instance.
Every are added into each classes if not already there, the class for the item will be added by 1 on its item.

For the prediction algorithn: CSV are loaded, then each item will be predicted by calculating the bayesian value of the item and take the highest probability classes as the prediction.
the data then will added into the naive data too.

### Prediction
| id    | age   | workclass        | education    | marital-status     | occupation      | relationship  | hours-per-week | income | 
|-------|-------|------------------|--------------|--------------------|-----------------|---------------|----------------|--------| 
| 26027 | young | Private          | HS-grad      | Never-married      | Craft-repair    | Not-in-family | normal         | <=50K  | 
| 26314 | young | Private          | Bachelors    | Divorced           | Exec-managerial | Not-in-family | normal         | <=50K  | 
| 31405 | young | Private          | Bachelors    | Married-civ-spouse | Prof-specialty  | Husband       | normal         | >50K   | 
| 14736 | adult | Private          | Some-college | Divorced           | Prof-specialty  | Not-in-family | normal         | <=50K  | 
| 27217 | young | Private          | HS-grad      | Married-civ-spouse | Exec-managerial | Husband       | many           | >50K   | 
| 5951  | young | Private          | Bachelors    | Never-married      | Prof-specialty  | Not-in-family | normal         | >50K   | 
| 30067 | young | Local-gov        | Bachelors    | Never-married      | Craft-repair    | Not-in-family | normal         | <=50K  | 
| 28777 | young | Self-emp-not-inc | Some-college | Never-married      | Craft-repair    | Not-in-family | normal         | <=50K  | 
| 15390 | adult | Private          | Some-college | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
| 18042 | young | Private          | Some-college | Married-civ-spouse | Exec-managerial | Husband       | normal         | >50K   | 
| 5793  | adult | Local-gov        | HS-grad      | Married-civ-spouse | Exec-managerial | Husband       | normal         | >50K   | 
| 31274 | adult | Private          | HS-grad      | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
| 17068 | young | Private          | HS-grad      | Married-civ-spouse | Craft-repair    | Husband       | low            | <=50K  | 
| 21894 | young | Private          | Bachelors    | Married-civ-spouse | Prof-specialty  | Husband       | normal         | >50K   | 
| 24128 | adult | Private          | Bachelors    | Married-civ-spouse | Exec-managerial | Husband       | normal         | >50K   | 
| 8550  | young | Private          | Bachelors    | Married-civ-spouse | Prof-specialty  | Husband       | normal         | >50K   | 
| 1181  | young | Private          | Bachelors    | Divorced           | Exec-managerial | Not-in-family | normal         | <=50K  | 
| 11149 | adult | Private          | HS-grad      | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
| 20836 | young | Private          | Some-college | Never-married      | Prof-specialty  | Not-in-family | normal         | <=50K  | 
| 25766 | adult | Local-gov        | Bachelors    | Married-civ-spouse | Prof-specialty  | Husband       | normal         | >50K   | 
| 139   | adult | Private          | Some-college | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
| 27160 | young | Private          | Bachelors    | Married-civ-spouse | Exec-managerial | Husband       | normal         | >50K   | 
| 8814  | young | Private          | HS-grad      | Married-civ-spouse | Exec-managerial | Husband       | normal         | >50K   | 
| 11470 | young | Private          | Bachelors    | Married-civ-spouse | Exec-managerial | Husband       | normal         | >50K   | 
| 16694 | adult | Private          | HS-grad      | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
| 26906 | adult | Private          | Bachelors    | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
| 9890  | adult | Private          | Some-college | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
| 9769  | adult | Private          | Bachelors    | Married-civ-spouse | Prof-specialty  | Husband       | normal         | >50K   | 
| 11078 | adult | Private          | Bachelors    | Married-civ-spouse | Exec-managerial | Husband       | normal         | >50K   | 
| 12924 | young | Private          | Bachelors    | Never-married      | Craft-repair    | Not-in-family | many           | <=50K  | 
| 26020 | adult | Private          | HS-grad      | Never-married      | Craft-repair    | Not-in-family | normal         | <=50K  | 
| 9017  | young | Private          | Some-college | Divorced           | Exec-managerial | Not-in-family | normal         | <=50K  | 
| 10243 | adult | Self-emp-not-inc | Some-college | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
| 10882 | young | Private          | HS-grad      | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
| 19535 | young | Self-emp-not-inc | HS-grad      | Never-married      | Craft-repair    | Not-in-family | normal         | <=50K  | 
| 4565  | adult | Private          | Bachelors    | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
| 29102 | young | Private          | Some-college | Never-married      | Exec-managerial | Not-in-family | normal         | <=50K  | 
| 26673 | adult | Private          | HS-grad      | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
| 29415 | adult | Private          | HS-grad      | Married-civ-spouse | Prof-specialty  | Husband       | normal         | >50K   | 
| 17397 | adult | Private          | Some-college | Married-civ-spouse | Craft-repair    | Husband       | normal         | >50K   | 
