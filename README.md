##Supervised Prediction of High Value Retail E-Commerce Customers Using Survival and LTV Analyses##

###Introduction and Usage:
* The combination of the programs below produces a survival analysis and Lifetime Value (LTV) analysis of retail e-commerce customers. After getting LTV values for each customer, supervised learning techniques are used to predict high-value customers based on user and behavioral attributes. The programs were built around the business model that a user can "save" a third-party retail item she/he likes to a database and can later purchase the product when it goes on sale. The retail ecommerce company then receives a commission from that sale.
* To conduct the full survival analysis, Lifetime Value (LTV) analysis, and prediction of high value customers, the programs should be run in sequential order as laid out below.

###frequencies.py

* __Main purpose__: To calculate the mean and median frequencies of use in days (and standard deviation of the usage). For instance, someone who has a mean frequency of use of 7 days and a standard deviation of 0 days can be thought of as having used the product consistently once every week on the same day of the week.

* __Input__: A .csv file with the following columns:

    * __id__: This identifies the instance when the item was saved.
    * __created_on__: The date and time when the item was saved.
    * __user_id__: Identifies the user who saved the item.


* __Output__: A .csv file with the following columns:
    * user_id, mean_freq, median_freq, std_freq, first_use_date, last_use_date, use_count

###purchase_info.py

* __Main purpose__: To summarize the purchasing habits at a user level.

* __Input__: A .csv file with the following columns:

    * __id__: This identifies the instance when a saved item was purchased.
    * __use_id__: This identifies the instance when the item was saved.
    * __user_id__: Identifies the user who saved the item.
    * __store_id__: Third-party retailer from which the item was purchased.
    * __transaction_date__: The date and time when the item was purchased.
    * __num_items__: Number of items purchased.
    * __total_order_value__: Value at which the user purchased the item.
    * __commission_value__: The commission received from the purchase.
    * __currency__: Currency of the value at which the user purchased the item.

* __Output__: 

    * user_id, num_items_purch, total_order_value, commission_value, first_purchase_amount, last_purch_date, first_purch_date, most_used_store

###combine_freq_purch_info.py
* __Main purpose__: To combine the output from frequencies.py with the output from purchase_info.py at a user level.

###survival_analysis.py
* __Main purpose__: To conduct a survival analysis using the Kaplan–Meier estimator and the lifelines package (https://github.com/CamDavidsonPilon/lifelines).

###lifetime_value.py
* __Main purpose__: To conduct a customer Lifetime Value (LTV) analysis using the results from the survival analysis.

###model_pred.py
* __Main purpose__: Prediction of high-value customers (using LTV labels) based on user attributes and behaviors such as amount of first purchase and time between first use and first purchase.

###plotting.py
* __Main purpose__: Some plotting of survival functions, LTVs, and use of product count histograms.

