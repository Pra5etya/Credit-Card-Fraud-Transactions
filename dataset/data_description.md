Problem Statement:

Company ABC, a major credit card company, faces challenges with their existing fraud detection system. The current system exhibits slow responsiveness in recognizing new patterns of fraud, leading to significant financial losses. To address this issue, they have contracted us to design and implement an algorithm that can efficiently identify and flag potentially fraudulent transactions for further investigation. The data provided consists of two tables: "cc_info," containing general credit card and cardholder information, and "transactions," containing details of credit card transactions that occurred between August 1st and October 30th.

Objective:
The primary goal of this project is to build an advanced fraud detection system using neural networks to identify transactions that appear unusual and potentially fraudulent. By applying object-oriented programming (OOPs) concepts, we aim to develop a scalable and modular solution that can handle large volumes of data and provide valuable insights to Company ABC.

Data Dictionary

We have two files in our dataset cc_info.csv and transactions.csv

Here is the column description for cc_info.csv

COLUMN NAME	DESCRIPTION
credit_card	Unique identifier for each transaction.
city	The city where the transaction occurred
state	The state or region where the transaction occurred
zipcode	The postal code of the transaction location
credit_card_limit	The credit limit associated with the credit card used in the transaction
Here is the column description for transactions.csv

COLUMN NAME	DESCRIPTION
credit_card	Unique identifier for each transactions
date	The date of the transaction (between August 1st and October 30th)
transaction_dollar_amount	The dollar amount of the transaction
Long	The longitude coordinate of the transaction location
Lat	The latitude coordinate of the transaction location
Lat	The credit limit associated with the credit card used in the transaction