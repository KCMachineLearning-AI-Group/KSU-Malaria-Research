# KSU-Malaria-Research
KC AI Lab partnership with KSU Research on the Open Source Malaria Project

## Describe problem statement


## Performance evaluation and considerations


## Various methods tried


## Selected approach



**Correlation Grouped – Mixed Step-wise Selection Algorithm:**

A problem with feature selection is the unexpected change in performance when making 
multiple moves at once. However, exhaustive search is too resource intensive for datasets that 
have thousands of columns. By grouping highly correlated features we were able to more efficiently 
search the feature space. Subsets of the feature space were tested for performance improvements from single 
feature changes.

### Feature elimination

Before running the mixed selection algorithm, features were removed that failed the distribution test, had 3 or 
fewer unique values, or had a variance of 0.

The distribution test identified features for which the training set and test set had significantly different 
values. In order to avoid extrapolating beyond the scope of the data available in the training set, these 
features were removed. A Kolmogorov-Smirnov goodness of fit test was run on each feature to compare the values 
in the training set to the values in the test set. If the test determined that there is enough evidence to 
conclude at 10% significance that the training set and test set come from a different distribution, the feature 
was removed before running the feature selection algorithm.

### Feature engineering

After features were removed using method above, all two-term feature interactions were tested for statistical 
significance using a linear regression model on the full train setm. Due to high-dimensionality, the 
interaction terms fed into the algorithm were first narrowed down to only interactions that had a statistically 
significant (p < 0.01) relationship to the IC50 value, controlling for the values of the two features of the 
interaction term.


### Mix-stepwise Algorithm

Algorithm:

`Group features into highly (99%) correlated buckets`

`Select random subset of features`

`Calculate benchmark model performance`

`Iterations without improvement = 0`

`Batch size increase as iterations without improvement increases`

`Loop`

    `If i is even:`

        `n = batch size + scaling factor`

        `Select n random features from current selected feature space`
	    
        `if largest RMSE performance gain > 0`
                
            'remove worst feature`
    
	`If i is odd:`
	
	    `n = batch size + scaling factor`
	    
        `Select a random feature from n random correlation groups to test for removal`
	
	    
	If any improve the score over the current benchmark, add the one with the largest improvement
	Establish new benchmark
	If benchmark improved, iter w/o improvement = 0, else +1
	If iter w/o improvement > 15, stop


## Test predictions and confidence interval analysis


## Feature importance analysis


## Recommendations and further analysis




## Model Validation
* [See trello for class documentation](https://trello.com/c/905HuiRU)
* See leaderboard.py for example usage
