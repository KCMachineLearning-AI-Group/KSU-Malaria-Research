# KSU-Malaria-Research
KC AI Lab partnership with KSU Research on the Open Source Malaria Project

## Project Goals

Summary
* Value add: Create efficiencies in testing compounds potent against malaria
* Test results from 47 different compounds as the target
* Chemical descriptors make up the independent variables
* 23 compounds remain untested and need accurately predicted test results
* More test data may be available in the future
*This area of research is called QSAR (quantitative structure-activity relationships)

Project Goals: 
* Develop model that can accurately identify new compounds with IC50<2
* Identify the features that are most likely to predict potency and explain adaptations that would increase potency

## Performance evaluation and considerations

### Model Validation
* Repeated stratified k-fold with 3 splits and 10 repeats
* Each split contained a single potent compound (IC50<=2)


## Various methods applied

Validation
* Repeated Stratified K-Fold adapted for regression
* Used all regression metrics available in sklearn
* Primarily RMSE for performance comparison

Applied Methods
* Stepwise Feature Selection
* Various additive models: linear regression, lasso, ridge
* Ensembles: RandomForest, AdaBoostRegressor, GradientBoostingRegressor
* SVMs: LinearSVR, SVR
* Feature Interactions
* Feature Elimination

Top Performing Methods
* Mixed-Stepwise Feature Selection
* Linear Support Vector Regressor
* Feature Interactions
* Distribution Test Feature Elimination

## Selected approach

A problem with feature selection is the unexpected change in performance when making 
multiple moves at once. However, exhaustive search is too resource intensive for datasets that 
have thousands of columns. By grouping highly correlated features we were able to more efficiently 
search the feature space. Subsets of the feature space were tested for performance improvements from single 
feature changes.

There is a high risk that any single high performing model has overfit the small training set, even when using
stratified k-fold cross validation. A robustness approach was taken which produced many predictions for each test
example and also produced coefficients for each feature for further analysis.

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

The step-wise algorithm was ran over 80 times, with random starting features and counts.

Algorithm:

    Group features into highly (99%) correlated buckets
    
    Select random subset of features
    
    Calculate benchmark model performance
    
    Iterations without improvement = 0
    
    Batch size increase as iterations without improvement increases
    
    Loop
    
        If i is even:
    
            n = batch size + scaling factor
    
            Select n random features from current selected feature space
            
            if largest RMSE performance gain > 0:
                remove worst feature
        
        If i is odd:
        
            n = batch size + scaling factor
            
            Select a random feature from n random correlation groups to test for removal
            
            if largest RMSE performance gain > 0:
                add best feature
	    
	    Set new benchmark
	
	    If benchmark improved:
	        iter w/o improvement = 0
        else: 
            iter w/o improvement++
	
	    If iter w/o improvement > 15:
	        stop


## Test predictions and confidence interval analysis

The predictions of the 80 models were considered for arriving at a 99% confidence interval for each of the compounds. 
A few compounds (OSM-S-146, OSM-S-151, OSM-S-152, OSM-S-153), had predictions ranging from large negative values to 
large positive values, which we believe could be safely ignored as not potent compounds. OSM-S-144 had the confidence 
interval closest to 0 value and is the most promising among the 23 compounds. Two other compounds which had prediction 
closer to minimum value were OSM-S-169 and OSM-S-170. The rest of the compounds had prediction values either starting 
from a double digit positive or a double digit negative numbers. Hence, based on the multiple runs,  OSM-S-144 had the 
confidence interval closest to minimum value (0.36) and, OSM-S-169 and OSM-S-170 were the next 2 compounds with closer 
to the minimum value. Another observation made from the predictions were, the runs which had predictions closer to 
minimum value included features ranging in values 75 - 95, while the converse of it is not true ( Not all runs which 
included features in that value range predicted less potency value). See [this visual](https://kate-young.github.io/KSUMalaria_Visualizations/) for the full test results.

## Feature importance analysis

Feature importance was measured in two ways, both the average coefficient for the feature and the percentage of times that
feature was selected when using the algorithm above. For instance, `AATSC0i` was selected in 88% of the final models
and had the smalled coefficient average of -0.294. This would indicate that a larger value at least strongly correlates
with a lower IC50 value. The next descriptor to consider is `ATSC3s` which had a large coefficient average of 0.28. 
This would indicate that a smaller descriptor value would corellate with an increase in potency. 
[This visual](https://kate-young.github.io/KSUMalaria_Visualizations/features.html) outlines all of the features and their measured importances.

## Recommendations and further analysis

This research alone may not be comprehensive enough to determine the next compound for testing. However, our recommendation
for which compounds to investigate further for potency against malaria would be first, OSM-S-144, and second OSM-S-169
and OSM-S-170. Our research highlighted those compounds as the most likely to have a potence against malaria.

Synthesizing compounds may also be an option for creating a potent compound that was not available in our test datast. 
Based on this research we would recommend exploring the impacts that a larger `AATSC0i` and a smaller `ATSC3s` would 
have on compound potency.




