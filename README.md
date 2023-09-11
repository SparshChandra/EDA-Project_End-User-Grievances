# EDA-Project_End-User-Grievances
Exploratory Data Analysis of End Users, Categorizing each complaint component using Machine Learning Algorithms 

# ABSTRACT

## Problem Statement

* The Consumer Financial Protection Bureau receives thousands of complaints from customers regarding financial products and services every week, and then it relays those complaints to the relevant companies. Using a classification algorithm can ensure that we classify those consumer complaints into the product category it belongs to using the description of the complaint.

## Objective

* Consumer complaints about financial services and products are actual complaints about financial services and products. This is a supervised text classification issue because each complaint is associated with a labeled product.
* This project's objective is to assign the complaint to a certain product category. It becomes a multiclass classification since it has numerous categories, which can be resolved by many machine learning techniques.
* Once the algorithm is in place, we can quickly classify any new complaints and then direct them to the appropriate party. Because we are reducing the need for human interaction to determine who should receive this complaint, this will save a lot of time.

## Use Cases

* A customer service department that wants to categorize the complaints it receives from clients would find this kind of model to be quite helpful. The department will be able to offer each group of consumers specialized solutions as a result of the department's classification of the problems it has received into buckets.
* This model can also be developed into a system that will suggest automatic solutions as new complaints are received in the future. These types of jobs were formerly carried out manually by numerous staff, which took a long time to complete and delayed a prompt response to concerns.

# DATASET COLLECTION

* The Consumer Complaint Database is a collection of complaints about various consumer financial products and services that we forwarded to various companies to acquire responses. Complaints are made public either 15 days after the customer receives a response from the corporation confirming a commercial relationship with them or 15 days after the consumer receives a response from the corporation confirming a business relationship with them, whichever comes first. Complaints submitted to other agencies, such as concerns about depository institutions with less than $10 billion in assets, are not included in the Consumer Complaint Database. Every day, the database is routinely updated.


# About the dataset

### Raw Data
* Number of columns - 19
* Number of rows - 1025010
* Target Feature - ‘Consumer Complaint’

### Data after EDA
* Number of columns - 02
* Number of rows - 277814
* Columns dropped - 17


### Data distribution of test and training

* 25% for test
* 75% for training
* Rows having missing data / Null Values – 747196

## Number of rows and columns
![Screenshot 2023-09-11 at 5 08 21 PM](https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/bee80d21-0dfa-4a93-9a95-e5cbe95618ab)

## Head of our dataset

* From this head of the dataset you can see the various columns of our dataset like date received, Product, issue, sub-product, company, state, zip code, tags, public response, etc.

![Screenshot 2023-09-11 at 5 09 24 PM](https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/4e75ed18-f415-4241-8d13-86d754c2a17e)

# DATA EXPLORATION / INSIGHTS

## Insights from the dataset:

![Screenshot 2023-09-11 at 5 10 10 PM](https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/2b671de3-e229-4e58-893d-0ea76ae14a93)

* From this graph you can see that most of the complaints are from the Debt collection category, followed by Mortgage, Credit reporting form and Credit card and the least are from Virtual currency and Other financial services.

## EDA -In Brief

* The dataset contains about a million records, all of which consist of 19 columns. We are only interested in products and information related to consumer complaints. The dataset is then refined by removing all columns except for Consumer Complaint and Product.

* Data purification, also known as pretreatment, is a common step in any project involving a large dataset. We'll show you an example of that in action using this data set. The consumer Complaint column in 747196 records has just blanks.

* To clean up our data, we intend to eliminate any records with a null value. With the dropna() function, nulls can be deleted.

* The Consumer complaint column had 747176 null values hence we dropped those rows.

![Screenshot 2023-09-11 at 5 12 20 PM](https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/7929aea2-44f0-44f3-ac14-a929e4e632e6)

* Now we have 277814 rows and none of the values are null.

![Screenshot 2023-09-11 at 5 12 49 PM](https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/a18645d6-0064-475c-b27b-66843ab71a37)

* We will only use the first 100,000 records in our dataset for model training and testing due to processing limitations.

* We create a new column category ID to represent each class as a numeric identifier, allowing our prediction model to comprehend the numerous categories more thoroughly.


## Text Preprocessing and Feature Engineering

* Text preprocessing is required in order for the algorithms to be able to predict.

* The Term Frequency - Inverse Document Frequency (TFIDF) weight will be utilized in this instance to determine how significant a word is to a document within a collection of documents.

![Screenshot 2023-09-11 at 5 14 32 PM](https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/e171a521-686a-4d34-a5de-c37ee6b718b5)

# TF-IDF Vectorization

* Following the removal of punctuation and lower-casing of the words, the frequency of a word is used to determine its relevance. The sum of the term's Term Frequency and Inverse Document Frequency scores is known as Term Frequency-Inverse Document Frequency or TF-IDF.
* TF - IDF = TF / IDF
* Term Frequency: This gives an overview of how frequently a word or phrase appears in a document.
* TF = Number of occurrences of the term in the document Overall Word count for the document.
* Words that frequently appear in documents are scaled down using the inverse document frequency. If a term appears in a small number of docs, it has a high IDF score. Likewise, a
The phrase would receive a low IDF score if it appears frequently in documents, for example, words like "the," "a," or "is".
* Word frequency scores called TF-IDF attempt to draw attention to words that are more intriguing, such as those that are common within a document but not across papers.
* The rarity of the phrase increases with the TFIDF score. For instance, the word "mortgage" would appear quite frequently in a complaint about mortgages. Mortgage would not likely be a common concern, though, if we looked at other grievances. We may deduce that, in comparison to other goods, the phrase "mortgage" is most likely a key one in mortgage complaints. Consequently, the mortgage would have a high TF-IDF score for complaints against the mortgage.

<img width="1003" alt="Screenshot 2023-09-11 at 5 17 40 PM" src="https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/3a4cee5e-8c72-4b08-be3f-594884a5f65e">

* We transform into numerical representations the written input of customer complaints and the textual output of product categories. Models of machine learning can only
process numerical data, thus this is required.
* We employ the Tfidf Vectorizer class from the sklearn.feature extraction.text package for this method.
* The process also involves first converting the input text to lowercase and then removing any special characters or single characters that may be present. These actions are executed by the clean text() function which we have defined in the code.
* To help our prediction engine make sense of all the different groups, we've included a new column labeled "category id."

# CLASSIFICATION MODELS

## K-Nearest Neighbor Classifier

* K-Nearest Neighbors (KNN) is an unstructured and time-saving machine-learning technique. To be non-parametric, there must be no presumption about the distribution of the underlying data. That is the form taken by the model after analysis of the data. In practice, when most datasets in the actual world do not conform to mathematical theoretical assumptions, this will be incredibly useful. To generate a model, a "lazy" algorithm doesn't require any training data. All data from the training phase were used in the test phase. As a result, training can be completed in less time and without adding extra expenses for testing. Time and mental energy are wasted during the testing process. The worst-case scenario for KNN is that it takes longer to scan all data points and more storage space is needed for the training data.

* Using KNN it got me the Accuracy of ~ 71%

## Linear SVC Model

* The Linear Support Vector Classifier (SVC) method applies a linear kernel function to perform classification and it performs well with a large number of samples. If we compare it with the SVC model, the Linear SVC has additional parameters such as penalty normalization which applies 'L1' or 'L2' and loss function. The kernel method cannot be changed in linear SVC, because it is based on the kernel linear method.

* Using Linear SVC, I achieved an Accuracy of ~ 79%

<img width="1001" alt="Screenshot 2023-09-11 at 5 21 29 PM" src="https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/07db42df-14ad-44f8-bc1b-3d10b17cb1a8">

<img width="999" alt="Screenshot 2023-09-11 at 5 22 05 PM" src="https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/1173baec-46a5-4fc0-8eb2-5989a406a417">

<img width="413" alt="Screenshot 2023-09-11 at 5 22 23 PM" src="https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/0654c404-2d94-4fa2-a81b-305df9730b47">

## CLASSIFICATION REPORTS

### Classification report for KNN

![Screenshot 2023-09-11 at 5 23 39 PM](https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/27575a6b-c450-41a6-bbfc-52ad441c53fe)

### Classification report for Linear SVC

![Screenshot 2023-09-11 at 5 24 04 PM](https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/5d011d86-ceab-42f6-a419-e5c2e90b584f)


## Confusion Matrix 

![Screenshot 2023-09-11 at 5 24 44 PM](https://github.com/SparshChandra/EDA-Project_End-User-Grievances/assets/102770866/55508796-9ec1-4c59-b086-533a794dbd04)

