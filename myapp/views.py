#*****************Importing all the neccessary Libraries**********************


from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import viewsets
from .models import *
from .serializers import *
import os
from rest_framework.parsers import MultiPartParser
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from easyocr import Reader
import argparse
import cv2
from django.shortcuts import redirect
from .models import Url
from .serializers import UrlSerializer
from rest_framework import status
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
import numpy as np
import gc
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import ExtraTreesClassifier
from urllib.parse import urlparse
import tldextract
from sklearn.model_selection import RandomizedSearchCV
import re
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework import status
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
import pickle
import requests
from bs4 import BeautifulSoup
from django.urls import path





# ******************Function and classes for sending and receiving data from extension**************
    
def new(request):
    return HttpResponse("Hello welcome! :)")
class SendStringdiscountAPIView(APIView):
    def get(self, request, format=None):

        """This Function fetches the current URL which user is visiting from the database 
        and sends the String which contains the information related to the Fake urgency"""
        
        last_website = Url.objects.latest('id')
        # last_website = url_name[0]
        website_url = last_website.url
        # sending url to the function which will scrape the data 
        website_text = scrape_website_text(website_url)
        # data is sent to the function which apply the ML algorithm
        data = detect_discounts(website_text)
        
        # serializer = CanonicSerializer(data, many=True)  # Serialize data
        return Response(data)
    

class SendMaliciousStringAPIView(APIView):
    def get(self, request, format=None):

        """This Function fetches the screenshot of the website user is visiting and
        then applies, then calls the function which will apply the ML algorithm and detect
        whether that website contains malicious links or not."""

        # Function to fetch the screenshot
        data_list = my_function()
        # Function which will apply ml algorithm to detect the malicious link.
        data = total_malicious(data_list)

        
        
        # serializer = CanonicSerializer(data, many=True)  # Serialize data
        return Response(data)
    

class SendforcedStringAPIView(APIView):
    def get(self, request, format=None):

        """The function Sends the information related to the Forced account creation"""
        
        data = forced_account()

        
        
        # serializer = CanonicSerializer(data, many=True)  # Serialize data
        return Response(data)


    
class SendStringfakereviewAPIView(APIView):
    def get(self, request, format=None):

        """It is used to detect the fake reviews in any E-commerce website.First it fetches the 
        URL of the website which the user is visiting from the database and then sends the url to
        the function which extract all the review string from that website and then sends that
        data to the function which will apply the ML algortihm to detect the fake reviews"""
        
        #fetching the last url
        last_website = Urlnew.objects.latest('id')
        # last_website = url_name[0]
        website_url = last_website.url
        # sending the url to the function which will extract the reviews using selenium
        website_text = detect_reviews(website_url)
        # function to detect fake reviews
        data = predict_spam_percentage(website_text)
        
        # serializer = CanonicSerializer(data, many=True)  # Serialize data
        return Response(data)
    

from rest_framework.decorators import api_view
@api_view(['POST'])
def receive_url_new(request):
    """This Function receives the url of the webpage in which user want to check the
    fake reviews"""
    serializer = UrlnewSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# *************************The following code is to save the screenshots**************************


# @csrf_exempt
class UploadScreenshot(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request, format=None):
        """This function is to save the Screenshot which extension is taking and save it"""
        if 'screenshot' not in request.FILES:
            return Response({'error': 'No file was submitted'}, status=status.HTTP_400_BAD_REQUEST)

        screenshot = request.FILES['screenshot']

        # Create a folder named 'screenshots' if it doesn't exist
        folder_path = os.path.join(os.getcwd(), 'screenshots')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Determine the file name by finding the next available screenshot number
        num = 1
        while os.path.exists(os.path.join(folder_path, f'screenshot{num}.png')):
            num += 1
        file_path = os.path.join(folder_path, f'screenshot{num}.png')

        # Write the uploaded file to the determined file path
        with open(file_path, 'wb') as screenshot_file:
            for chunk in screenshot.chunks():
                screenshot_file.write(chunk)

        return Response({'success': f'Screenshot saved as screenshot{num}.png'}, status=status.HTTP_201_CREATED)
    

#********************This code is to fetch url from website*****************************




class FetchURLView(APIView):
    def post(self, request):
        url = request.data.get('url')
        print("Received URL:", url)
        return Response({"message": "URL received"})
    

from rest_framework.decorators import api_view


@api_view(['POST'])
def receive_url(request):
    """This Function fetch the url the webpage of which we want to check the fake urgency"""
    serializer = UrlSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#********************from api wala***********************




class WebsiteView(APIView):
    def post(self, request, format=None):
        serializer = WebsiteSerializer(data=request.data)

        if serializer.is_valid():
            # url_name.append(serializer.data)
            serializer.save()
            

            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    


class SendStringAPIView(APIView):
    def get(self, request, format=None):

        """This function sends the percentage of the dark pattern present in the websiite 
        which user is visiting"""
        
        last_website = Website.objects.latest('id')
        # last_website = url_name[0]
        website_url = last_website.url
        # url = fetchurl()
        data = predict(website_url)
        # predict(last_website)
        # data = fetchdata()
        # data = string_data[0]
        
        # serializer = CanonicSerializer(data, many=True)  # Serialize data
        return Response(data)
    

class TagListViewSet(viewsets.ModelViewSet):
    queryset = TagList.objects.all()
    serializer_class = TagListSerializer
    
class TagListView(APIView):
    def get(self, request, format=None):
        """This Sends the list of tags to the extension which contains the dark pattern
        so that extension creates the bounding box arround the dark pattern elements."""
        last_website = Website.objects.latest('id')
        website_url = last_website.url
        tag_list = predict_id(website_url)
        # tag_list = ['div._3_L3jD', 'body','_1mXcCf RmoJUa']  # Your list of strings
        return Response(tag_list)
    


class DetailedListView(APIView):
    def get(self, request, format=None):
        """This Function Sends all the dark pattern information to the extension in the form of list.it calls
        all the functions and appends its output in the list."""
        data_list = []
        # ********Percentage***********
        last_website = Website.objects.latest('id')
        website_url = last_website.url
        percent = predict(website_url)
        percentage = (f"{percent}% Dark Pattern Found In this Website")
        data_list.append(percentage)

        # *******Fake Discount***********
        last_website1 = Url.objects.latest('id')
        website_url1 = last_website1.url
        website_text1 = scrape_website_text(website_url1)
        data = detect_discounts(website_text1)
        data_list.append(data)

        #*******Forced account**********
        data1 = forced_account()
        data_list.append(data1)

        #********Fake Review************

        last_website2 = Urlnew.objects.latest('id')
        website_url2 = last_website2.url
        website_text2 = detect_reviews(website_url2)

        data2 = predict_spam_percentage(website_text2)
        data_list.append(data2)

        #***********Malicious link***************

        data_list1 = my_function()
        data3 = total_malicious(data_list1)
        data_list.append(data3)
        data_listnew = np.array(data_list)

        return Response(data_list)
    


#**********************Code to fetch the screenshot from folder******************************
def get_last_screenshot():
    """This Function is to get the path of last saved screenshot path so that it can be used 
    for further analysis."""
    screenshots_dir = os.path.join(settings.BASE_DIR, 'screenshots')  # Assuming screenshots are stored in BASE_DIR/screenshots
    if not os.path.exists(screenshots_dir):
        # Handle the case where the directory doesn't exist
        return None
    
    screenshots = [f for f in os.listdir(screenshots_dir) if f.startswith('screenshot')]
    if not screenshots:
        # Handle the case where there are no screenshots
        return None

    # Sort the screenshots based on modification time
    screenshots.sort(key=lambda x: os.path.getmtime(os.path.join(screenshots_dir, x)))
    
    # Get the last (most recent) screenshot
    last_screenshot = screenshots[-1]
    
    return os.path.join(screenshots_dir, last_screenshot)




def my_function():
    """This function is used to extract the strings from the image."""
    last_screenshot_path = get_last_screenshot()
    
    if last_screenshot_path is None:
        print("No screenshots found.")
    
    
    reader = Reader(['en'])
    result = reader.readtext(last_screenshot_path,paragraph=True)
    data_list = []
    for i in result:
    
        data_list.append(i[1])
    print("Last screenshot path:", last_screenshot_path)
    return data_list
    # Now you can use last_screenshot_path as needed
    

#******************code to extract links**********************

def extract_links(texts):
    # Regular expression to match URLs
    """This function is used to filter out the links from the list of strings.Regular Expression
    is used to find the links."""
    pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)" \
              r"(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+" \
              r"(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

    url_pattern = re.compile(pattern)

    # List to store extracted links
    links = []

    for text in texts:
        # Find all matches in the current string
        matches = url_pattern.findall(text)
        # Add all matches to the links list
        links.extend([match[0] for match in matches])

    return links





def admin_redirect(request):
    """This function is for the testing of the differnt functions."""
    # my_function()
    
    return redirect('/admin/')


#**********************Code to Scrape the website using selenium***********************

def scrape_website_text(url):
    # Initialize Chrome webdriver
    driver = webdriver.Chrome()
    # Open the URL
    driver.get(url)
    # Wait for the page to load (adjust the sleep time as needed)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    # Get all elements on the page
    all_elements = driver.find_elements(By.XPATH, "//*")
    # Initialize a list to store all text content
    all_text = []
    # Iterate through all elements
    for element in all_elements:
        # Attempt to extract text content, handle StaleElementReferenceException
        try:
            text = element.text.strip()
            if text:
                all_text.append(text)
        except StaleElementReferenceException:
            pass  # Ignore the exception and continue with the next element
    # Close the webdriver
    driver.quit()
    return all_text



#************************Code to find the price***************************
def check_rupee(line):
    rupee_pattern = r'^₹'
    return re.search(rupee_pattern, line)


def print_rupee_lines(lines):
    for line_num, line in enumerate(lines, start=1):
        if check_rupee(line):
            price =line.replace(',', '')
            price1 = price[1:]
            final_price = int(price1)
            return final_price
            break

#**********************Code to detect the url*************************

def extract_links(texts):
    # Regular expression to match URLs
    pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)" \
              r"(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+" \
              r"(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

    url_pattern = re.compile(pattern)

    # List to store extracted links
    links = []

    for text in texts:
        # Find all matches in the current string
        matches = url_pattern.findall(text)
        # Add all matches to the links list
        links.extend([match[0] for match in matches])

    return links

#******************Code to check whther the link is malicious or not******************

def extract_url_features(url, words_list):
    # Parse the URL
    parsed_url = urlparse(url)

    # Domain Features
    domain = parsed_url.netloc
    domain_length = len(domain)
    num_subdomains = domain.count('.')
    has_hyphen = '-' in domain

    # Path Features
    path = parsed_url.path
    path_length = len(path)
    num_path_segments = path.count('/')

    # Top-Level Domain (TLD) Features
    tld_info = tldextract.extract(domain)
    tld = tld_info.suffix
    tld_length = len(tld)
    subdomain = tld_info.subdomain

    # Protocol Features
    protocol = parsed_url.scheme
    is_https = protocol == 'https'

    # Query Parameters Features
    query_params = parsed_url.query
    num_query_params = len(query_params.split('&'))

    # Define a fixed set of character features
    char_features = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~!*'()$&+,;=:/?#@[]{}|\\^")

    # Calculate character frequencies for the defined characters
    char_frequencies = {char: url.lower().count(char) for char in char_features}

    # Calculate keyword counts for the defined keywords
    keyword_counts = {keyword: url.lower().count(keyword) for keyword in words_list}

    # URL Length Features
    url_length = len(url)

    # URL Structure Features
    has_redirect = '->' in url
    has_shortened_url = re.search(r'bit\.ly|t\.co|ow\.ly', url) is not None

    # Create the fixed-length feature array
    feature_array = [
        domain_length, num_subdomains, int(has_hyphen),
        path_length, num_path_segments, tld_length,
        int(is_https), num_query_params, url_length,
        int(has_redirect), int(has_shortened_url)
    ]

    # Add character frequencies as features in a fixed order
    feature_array.extend([char_frequencies[char] for char in char_features])

    # Add keyword counts as features in a fixed order
    feature_array.extend([keyword_counts[keyword] for keyword in words_list])

    return feature_array



def preprocessing_url(url):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path_wordlist = os.path.join(base_dir, 'words_list.npy')
    common_keywords = np.load(file_path_wordlist)
    dataWithFeature1 = []
    features1 = extract_url_features(url, common_keywords)
    newRows1 = []
    for feature in features1:
        newRows1.append(feature)

    # Add the label at the end of newRow
#         newRows.append(label)

        dataWithFeature1.append(newRows1)

    lab = LabelEncoder()
    
    data_new2 = pd.DataFrame(dataWithFeature1)
    for i  in data_new2.select_dtypes(include="object").columns.values:
        data_new2[i]=lab.fit_transform(data_new2[i])
        
    X_test1 = data_new2.iloc[:, 0:-1]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path_model = os.path.join(base_dir, 'urlmodelnew.pkl')

# Open the file in binary read mode
    with open(file_path_model, 'rb') as f:
    # Load the saved model using pickle.load()
        model = pickle.load(f)
        
    X_test1 = data_new2.iloc[:, 0:-1]
    
    prediction1 = model.predict(X_test1)

    unique_elements, counts = np.unique(prediction1, return_counts=True)

    # Find the index of the most common element
    index_most_common = np.argmax(counts)

    # Get the most common element
    most_common_element = unique_elements[index_most_common]
    
    return most_common_element


#***********************************Code to find the malicious url********************

def total_malicious(texts):
    links = extract_links(texts)
    if(len(links)==0):
        out_put = "No Malicious Link detected"
        print(type(out_put))
        return out_put

    else:
        count_num = []
        for link in links:
            ans = preprocessing_url(link)
            if(ans == 'True'):

                count_num.append(ans)
        if(len(count_num)>0):
            out_put1 = (f"{len(count_num)} Malicious link detected")
            return out_put1
        else:
            out_put2 = "No Malicious Link Detected"
            return out_put2

    

#******************************Code to find Discount**********************************

def detect_discounts(strings):
    discount_patterns = [
        r'\d+% off',
        r'\$[\d.]+ off',
        r'save \$[\d.]+',
        r'discount of \$[\d.]+',
        r'few left',
        r'limited stock',
        r'only \d+ left',
        r'\d+ left in stock'
    ]
    
    discounts = []
    for text in strings:
        for pattern in discount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                discounts.append(match)
                
    if(len(discounts)>1):
        out_put = (f'{len(discounts)} Fake Urgency and Sarcity detected')
        return out_put
    else:
        out_put1 = "There is No Fake urgency and scarcity"
        return out_put1
    

#******************code to find the fake review****************************
    
import re

def detect_reviews(url):
    text_list = scrape_website_text(url)
    positive_words = ["good", "great", "excellent", "wonderful", "awesome", "amazing", "fantastic", "superb", "outstanding", "love", "loved", "like", "enjoy", "satisfied", "pleased", "impressed", "Fine", "High quality", "surprising", "stunning", "wonderful", "incredible", "majestic", "satisfactory", "splendid", "brilliant", "stellar", "splendid", "exceptional", "fabulous", "marvelous", "five star", "jhakas", "faadu", "bindaas", "Bhaut", "acha", "mja", "aagya", "sahi", "hai", "jarur", "buy kro"]
    negative_words = ["bad", "poor", "terrible", "awful", "horrible", "disappointing", "disappointed", "unhappy", "hate", "hated", "dislike", "regret", "waste", "wasted", "frustrating", "unpleasant", "inferior", "broken", "problem", "bug", "Unreliable", "issues", "difficult", "slow", "misleading", "Disagreeable", "unpleasant", "cheap", "lame", "frustrating"]

    review_patterns = []
    review_pattern1 = []
    review_list = []

    for text in text_list:
        # Compile regex patterns for positive and negative words
        positive_pattern = re.compile(r'\b(?:{})\b'.format('|'.join(positive_words)), flags=re.IGNORECASE)
        negative_pattern = re.compile(r'\b(?:{})\b'.format('|'.join(negative_words)), flags=re.IGNORECASE)

        # Search for matches
        positive_match = positive_pattern.search(text)
        negative_match = negative_pattern.search(text)

        # Only append if negative word found and no positive word found
        if negative_match and not positive_match:
            review_patterns.append(text)

        # Only append if positive word found and no negative word found
        if positive_match and not negative_match:
            review_pattern1.append(text)

        review_list = review_patterns + review_pattern1

    return review_list




def predict_spam_percentage(input_strings):
    # Load the SVM model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the base directory of your Django app

# Construct the file path relative to the base directory
    file_path_new44 = os.path.join(base_dir, 'svm_model.pkl')

    # file_path_new4 = path("myapp/new4.pkl")
    # file_pathvectorizer1 = path("myapp/vectorizer1.pkl")
    file_pathvectorizer12 = os.path.join(base_dir, 'tfidf_vectorizer.pkl')
    with open(file_path_new44, 'rb') as model_file:
        loaded_svm_model = pickle.load(model_file)

    # Load the TF-IDF vectorizer
    with open(file_pathvectorizer12, 'rb') as vectorizer_file:
        loaded_tfidf_vectorizer = pickle.load(vectorizer_file)

    # Transform input strings using the loaded TF-IDF vectorizer
    input_tfidf = loaded_tfidf_vectorizer.transform(input_strings)

    # Predict using the loaded SVM model
    predictions = loaded_svm_model.predict(input_tfidf)

    # Calculate percentage of spam strings
    spam_count = np.sum(predictions)
    total_count = len(predictions)
    spam_percentage = (spam_count / total_count) * 100
    
    if(spam_percentage < 30):
        out_put = "This Website contains True Reviews"
        return out_put
    else:
        out_put1 = "This Website Contains Fake Reviews.Please be alert While Shopping."
        return out_put1

def fetchdata():
    data = Canonic.objects.all().order_by("-id")
    new_data = data
    ans = str(new_data[0])
    
    
    return ans
# Ml model starts from here




def scrape_website(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all HTML elements with text and their corresponding IDs
        elements_with_ids = soup.find_all(lambda tag: tag.name and tag.text.strip() and tag.get('id'))

        # Extract text and ID information
        text_with_ids = [(element.get('id'), element.text.strip()) for element in elements_with_ids]

        return text_with_ids
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return None
    

def createdataframe(url_to_scrape):
    
#url_to_scrape = 'https://www.flipkart.com/realme-c51-mint-green-64-gb/p/itm0e93bcb87927f?pid=MOBGSQGGC7NY4PXC&lid=LSTMOBGSQGGC7NY4PXCOCBP4E&marketplace=FLIPKART&fm=neo%2Fmerchandising&iid=M_520fe8e5-3061-4da5-a7da-518459743db1_59_SRKS9OGSRSRY_MC.MOBGSQGGC7NY4PXC&ppt=clp&ppn=mobile-phones-store&ssid=cktbhw5yqo0000001707127179937&otracker=clp_pmu_v2_Under%2B%2B%25E2%2582%25B910%252C000_1_59.productCard.PMU_V2_realme%2BC51%2B%2528Mint%2BGreen%252C%2B64%2BGB%2529_mobile-phones-store_MOBGSQGGC7NY4PXC_neo%2Fmerchandising_0&otracker1=clp_pmu_v2_PINNED_neo%2Fmerchandising_Under%2B%2B%25E2%2582%25B910%252C000_LIST_productCard_cc_1_NA_view-all&cid=MOBGSQGGC7NY4PXC'
    result = scrape_website(url_to_scrape)

#if result:
    # Create a DataFrame from the extracted information
    df = pd.DataFrame(result, columns=['ID', 'Content'])

    # Save the DataFrame to a CSV file with an index
    #df_new = df.to_csv('output_table8.csv', index=False)
    
    return df



from django.conf import settings


def predict(url_to_scrape):
    

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the base directory of your Django app

# Construct the file path relative to the base directory
    file_path_new4 = os.path.join(base_dir, 'new4.pkl')

    # file_path_new4 = path("myapp/new4.pkl")
    # file_pathvectorizer1 = path("myapp/vectorizer1.pkl")
    file_pathvectorizer1 = os.path.join(base_dir, 'vectorizer1.pkl')
    with open(file_path_new4, 'rb') as file:
        loaded_model = pickle.load(file)
        if(loaded_model):
            print("All good")
    with open(file_pathvectorizer1, 'rb') as file1:
        count_vect = pickle.load(file1)
    data = createdataframe(url_to_scrape)

    
    #test1 = pd.read_csv('4.csv')
    test1 = data.drop('ID' , axis=1)
    tfidf_transformer = TfidfTransformer()
    new_data_counts = count_vect.transform(test1['Content'])
    new_data_tfidf = tfidf_transformer.fit_transform(new_data_counts)


    predictions = loaded_model.predict(new_data_tfidf)
    
    
    
   
    t1= pd.DataFrame({ "pattern": predictions})
    target_value = 'Dark'
    if target_value in t1['pattern'].values:
        t1.value_counts()['Dark']
        percentage = t1.value_counts()['Dark']*100/(t1.value_counts()['Dark']+t1.value_counts()['Not Dark'])
        #print(f"The website you are visiting have {percentage}% Dark Pattern")
        # string_data.append(percentage)
        percentagenew = round(percentage,2)
        # instance = Canonic.objects.create(name=percentagenew)
        return percentagenew
        
        
    
    else:
        # instance = Canonic.objects.create(name="5")
        ans = "0"
        return ans
        
# new code try
        


        



def predict_id(url_to_scrape):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path_new4 = os.path.join(base_dir, 'new4.pkl')
    file_pathvectorizer1 = os.path.join(base_dir, 'vectorizer1.pkl')
    with open(file_path_new4, 'rb') as file:
        loaded_model = pickle.load(file)
    with open(file_pathvectorizer1, 'rb') as file1:
        count_vect = pickle.load(file1)
    data = createdataframe(url_to_scrape)
    #test1 = pd.read_csv('4.csv')
    test1 = data.drop('ID' , axis=1)
    
    tfidf_transformer = TfidfTransformer()
    new_data_counts = count_vect.transform(test1['Content'])
    new_data_tfidf = tfidf_transformer.fit_transform(new_data_counts)


    predictions = loaded_model.predict(new_data_tfidf)
    
    
    
   
    t1= pd.DataFrame({ "pattern": predictions})
    indices_list = []

# Iterate through the DataFrame
    for index, value in t1['pattern'].items():
        if value == 'Dark':
            indices_list.append(index)
            
    selected_elements = ['#' + item for sublist in data.loc[indices_list, 'ID'].dropna().tolist() for item in (sublist if isinstance(sublist, list) else [sublist])]


    return selected_elements
    

#*************************code for forced signup*******************************

def find_keywords(text_list):
    keywords = ["signup", "login", "signin"]
    found_keywords = []

    for text in text_list:
        for keyword in keywords:
            if keyword in text.lower():
                found_keywords.append(keyword)

    # Remove duplicates
    found_keywords = list(set(found_keywords))
    
    return found_keywords


def forced_account():
    texts = my_function()
    list_text = find_keywords(texts)
    
    if(len(list_text)>0):
        out_put = "This Website Contains Forced account creation."
        return out_put
    else:
        out_put1 = "This Website does not contain any Forced account creation."
        return out_put1







    

