# RailCar-leasing
We need to predict lease rates for the two car types provided using industry railcar demand (orders, deliveries, &amp; backlog), carloads, and other exogenous features.

The Excel workbook contains three tabs, of which to use for model building purposes
1.	New Railcar Demand – North American Orders, Deliveries, & Backlog for the Overall Industry and two Railcar Types 
a.	Orders: The number of quarterly orders for new railcars by car type 
b.	Deliveries: The number of quarterly deliveries for new railcars by car type
c.	Backlog: The quarterly sum of railcars by car type in manufacturing backlogs.  The current Backlog can be estimated via: Prior Backlog + Current Orders – Current Deliveries
2.	Lease Rates – lease rates for the two railcar types (target variables, Covered Hoppers & Tankcars)
3.	Carloads – weekly carloads by end market (exogenous features for modeling).  Carloads are defined as the total number of loaded railcar shipments per week, categorized by commodity type/end market

A data modeling language used to build the models Python[QS - Problem Set Data - Confidential.xlsx](https://github.com/mahimaneema/RailCar-leasing/files/8904914/QS.-.Problem.Set.Data.-.Confidential.xlsx)
