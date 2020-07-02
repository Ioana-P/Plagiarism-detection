import re
import pandas as pd
import operator 
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk import RegexpTokenizer
import numpy as np


# Add 'datatype' column that indicates if the record is original wiki answer as 0, training data 1, test data 2, onto 
# the dataframe - uses stratified random sampling (with seed) to sample by task & plagiarism amount 

# Use function to label datatype for training 1 or test 2 
def create_datatype(df, train_value, test_value, datatype_var, compare_dfcolumn, operator_of_compare, value_of_compare,
                    sampling_number, sampling_seed):
    # Subsets dataframe by condition relating to statement built from:
    # 'compare_dfcolumn' 'operator_of_compare' 'value_of_compare'
    df_subset = df[operator_of_compare(df[compare_dfcolumn], value_of_compare)]
    df_subset = df_subset.drop(columns = [datatype_var])
    
    # Prints counts by task and compare_dfcolumn for subset df
    #print("\nCounts by Task & " + compare_dfcolumn + ":\n", df_subset.groupby(['Task', compare_dfcolumn]).size().reset_index(name="Counts") )
    
    # Sets all datatype to value for training for df_subset
    df_subset.loc[:, datatype_var] = train_value
    
    # Performs stratified random sample of subset dataframe to create new df with subset values 
    df_sampled = df_subset.groupby(['Task', compare_dfcolumn], group_keys=False).apply(lambda x: x.sample(min(len(x), sampling_number), random_state = sampling_seed))
    df_sampled = df_sampled.drop(columns = [datatype_var])
    # Sets all datatype to value for test_value for df_sampled
    df_sampled.loc[:, datatype_var] = test_value
    
    # Prints counts by compare_dfcolumn for selected sample
    #print("\nCounts by "+ compare_dfcolumn + ":\n", df_sampled.groupby([compare_dfcolumn]).size().reset_index(name="Counts") )
    #print("\nSampled DF:\n",df_sampled)
    
    # Labels all datatype_var column as train_value which will be overwritten to 
    # test_value in next for loop for all test cases chosen with stratified sample
    for index in df_sampled.index: 
        # Labels all datatype_var columns with test_value for straified test sample
        df_subset.loc[index, datatype_var] = test_value

    #print("\nSubset DF:\n",df_subset)
    # Adds test_value and train_value for all relevant data in main dataframe
    for index in df_subset.index:
        # Labels all datatype_var columns in df with train_value/test_value based upon 
        # stratified test sample and subset of df
        df.loc[index, datatype_var] = df_subset.loc[index, datatype_var]

    # returns nothing because dataframe df already altered 
    
def train_test_dataframe(clean_df, random_seed=100):
    
    new_df = clean_df.copy()

    # Initialize datatype as 0 initially for all records - after function 0 will remain only for original wiki answers
    new_df.loc[:,'Datatype'] = 0

    # Creates test & training datatypes for plagiarized answers (1,2,3)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.gt, 0, 1, random_seed)

    # Creates test & training datatypes for NON-plagiarized answers (0)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.eq, 0, 2, random_seed)
    
    # creating a dictionary of categorical:numerical mappings for plagiarsm categories
    mapping = {0:'orig', 1:'train', 2:'test'} 

    # traversing through dataframe and replacing categorical data
    new_df.Datatype = [mapping[item] for item in new_df.Datatype] 

    return new_df


# helper function for pre-processing text given a file
def process_file(file):
    # put text in all lower case letters 
    err_i=0
    try: 
        all_text = file.read().lower()
    except UnicodeDecodeError:
        all_text=''
        for line in file:
            line=line.strip()
            line=line.decode('utf-8','ignore').encode("utf-8")
            all_text +=' ' + line

    

    # remove all non-alphanumeric chars
    all_text = re.sub(r"[^a-zA-Z0-9]", " ", all_text)
    # remove newlines/tabs, etc. so it's easier to match phrases, later
    all_text = re.sub(r"\t", " ", all_text)
    all_text = re.sub(r"\n", " ", all_text)
    all_text = re.sub("  ", " ", all_text)
    all_text = re.sub("   ", " ", all_text)
    
    return all_text


def create_text_column(df, file_directory='data/'):
    '''Reads in the files, listed in a df and returns that df with an additional column, `Text`. 
       :param df: A dataframe of file information including a column for `File`
       :param file_directory: the main directory where files are stored
       :return: A dataframe with processed text '''
   
    # create copy to modify
    text_df = df.copy()
    
    # store processed text
    text = []
    
    # for each file (row) in the df, read in the file 
    for row_i in df.index:
        filename = df.iloc[row_i]['File']
        #print(filename)
        file_path = file_directory + filename
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:

            # standardize text using helper function
            file_text = process_file(file)
            # append processed text to list
            text.append(file_text)
    
    # add column to the copied dataframe
    text_df['Text'] = text
    
    return text_df


def numerical_dataframe(csv_file='data/file_information.csv'):
    '''Reads in a csv file which is assumed to have `File`, `Category` and `Task` columns.
       This function does two things: 
       1) converts `Category` column values to numerical values 
       2) Adds a new, numerical `Class` label column.
       The `Class` column will label plagiarized answers as 1 and non-plagiarized as 0.
       Source texts have a special label, -1.
       :param csv_file: The directory for the file_information.csv file
       :return: A dataframe with numerical categories and a new `Class` label column'''
    data = pd.read_csv(csv_file)
    data.loc[data.Category=='non']
    data.Category.replace({'non':0,
                           'heavy':1,
                          'light':2,
                          'cut':3,
                          'orig':-1}, inplace=True)
    data['Class'] = data['Category'].values
    data.Class.replace({2:1,
                          3:1}, inplace=True)
    
    # your code here
    
    return data

def reduce_to_one(x):
    if x>1:
        return 1
    else:
        return x


def calculate_containment(df, n, answer_filename):
    '''Calculates the containment between a given answer text and its associated source text.
       This function creates a count of ngrams (of a size, n) for each text file in our data.
       Then calculates the containment by finding the ngram count for a given answer text, 
       and its associated source text, and calculating the normalized intersection of those counts.
       :param df: A dataframe with columns,
           'File', 'Task', 'Category', 'Class', 'Text', and 'Datatype'
       :param n: An integer that defines the ngram size
       :param answer_filename: A filename for an answer text in the df, ex. 'g0pB_taskd.txt'
       :return: A single containment value that represents the similarity
           between an answer text and its source text.
    '''
    
    # your code here
    
    try:
        student_ans = process_file(open(answer_filename, 'r'))
    except FileNotFoundError:
        answer_filename = 'data/' + answer_filename
        student_ans = process_file(open(answer_filename, 'r'))
   

    task = answer_filename[-5]
    wiki_text_ans_loc = df.loc[((df['Task']==task) & (df['Category'] ==-1))]['File']
    wiki_text_ans_filename = wiki_text_ans_loc.iloc[0]
    
    try:
        wiki_ans = process_file(open(wiki_text_ans_filename, 'r'))
    except FileNotFoundError:
        wiki_text_ans_filename = 'data/' + wiki_text_ans_filename
        wiki_ans = process_file(open(wiki_text_ans_filename, 'r'))
   
    
    cvectorizer = CountVectorizer(ngram_range=(n,n))
    
    doc = cvectorizer.fit_transform([wiki_ans, student_ans])
    doc_df = pd.DataFrame(doc.todense(), columns=cvectorizer.get_feature_names())
    
    answer_vec = doc_df.iloc[1].values
    ngram_length_ans = np.sum(answer_vec)

    input_df_vect = doc_df.iloc[0].values
    comp_df_vect = doc_df.iloc[1].values
    
    containment_value = np.sum(np.minimum(input_df_vect,comp_df_vect))/np.sum(comp_df_vect)


    return containment_value