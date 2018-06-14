<h1>Streamlined Machine Learning</h1>
<hr>
<strong>Streamlined Machine Learning</strong> is a set of robust functions and classes meant to streamline: preprocessing, model selection, and feature selection.
This package is build on top of <em>SciPy</em> and <em>sklearn</em>.

<h2>Basic Usage</h2>
By building a <code>Stream</code> object, you can specify a list of predefined objects the package manages, then you can <code>flow</code> through them each on default grid selection parameters or user defined parameters (denoted <code>params</code>).
Streams provided:
<ul>

<li><code>TransformationStream</code>, meant to flow through preprocessing techniques such as: scaling, normalizing, boxcox, binarization, pca, or kmeans aimed at returning a desired input dataset for model development.</li>

<li><code>ModelSelectionStream</code>, meant to flow through several predictive models to determine which is the best, these include: LinearRegression, SupportVectorRegressor, RandomForestRegressor, KNNRegressor, and others. You must specify whether your steam is a <em>regressor</em> or <em>classifier</em> stream (denoted <code>regressor=True</code> and <code>classifier=True</code> </li>

<li><code>FeatureSelectionStream</code>, meant to flow through several predictive models and algorithms to determine which subset of features is most predictive or representative of your dataset, these include: RandomForestFeatureImportance, LassoFeatureImportance, MixedSelection, and a technique to ensemble each named TOPSISFeatureRanking. You must specify whether your wish to ensemble and with what technique (denoted <code>ensemble=True</code> </li>

<ul></ul>
</ul>

<hr>

<h2>Current Implementation</h2>

Currently we support transformation streams and restricted model selection streams with 5 regression estiminators.

Example of a transformation stream:

<code> formed = TransformationStream.flow(["scale","normalize","pca","binarize","kmeans"], 
                        params={"pca__percent_variance":0.75, "binarize__threshold":0.0, "kmeans__n_clusters":3}) 
</code>

<code>
  performances = ModelSelectionStream(X,y).flow(["svr", "lr", "knnr","lasso","abr"],
                                              params={'svr__C':[1,0.1,0.01,0.001],
                                                     'lr__fit_intercept':[False, True],
                                                     'knnr__n_neighbors':[5,10],
                                                     'lasso__alpha':[1.0,10.0,20.0],
                                                     'abr__n_estimators':[10,20,50],
                                                     'abr__learning_rate':[0.1,1,10]},
                                              verbose=True)
  </code>


