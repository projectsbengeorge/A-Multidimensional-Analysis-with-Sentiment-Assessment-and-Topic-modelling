#!/usr/bin/env python
# coding: utf-8

# In[1]:


import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import time
import csv



#Comments -- Variables used to extract the data from amazon website are declared globally --

broswer_open_flag = 0

model_list = []
orginal_product_rating_list = []
orginal_product_rating_count_list = []
company_list = []
review_title_list = []
reviewer_name_list = []
review_rating_list = []
review_list = []
review_helpful_rating_list = []


# In[2]:


#Comments -- Reading the config CSV file --

def config_file_reader_method():
    
    open_csv = open('D:/amazon_review_data/Config_file.csv','r') 
    config_file_reader = csv.reader(open_csv,delimiter=',')
    config_file_data = list(zip(*config_file_reader))
    open_csv.close() # closing the config file
    return config_file_data


# In[3]:


#Comments -- A method is defined to open chrome browser using selenium webdriver

def selenium_chrome_driver_method(url):
    
    global broswer_open_flag
    global driver

    if broswer_open_flag == 0:
        driver = webdriver.Chrome()
        driver.get(url)
        driver.refresh()
     
       
        broswer_open_flag = 1
        
    else:
        
      
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(url)

        driver.switch_to.window(driver.window_handles[0])
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
    
    return driver
    


# In[4]:



#Comments -- A method is defined to scrape the data from amazon reviews tab using Selenium --

def review_scraper_method(driver):
    
    
    #Comments -- Checking if any foreign language comments are avilabe in the review page
    Xpath_review_title,Xpath_reviewer_name,Xpath_review_rating = foreign_lang_comments_check_method()

      
    
    #Comments -- The main element of the review page is stored into the variable 'all_reviews'
    
    all_reviews = driver.find_elements(By.XPATH,"//div[@class='a-section review aok-relative']")
    for all_review in all_reviews:
        
  

       #Comments -- fetching overall product data like Model,company name,overall product ratings --

        model_names = driver.find_elements(By.XPATH,"//span[@class='a-list-item']/a[@class='a-link-normal']")
        for model in model_names:
            model_list.append(model.text)

        company_names = driver.find_elements(By.XPATH,"//span[@id='cr-arp-byline']/a[@class='a-size-base a-link-normal']")
        for company in company_names:
            company_list.append(company.text)


        orginal_product_ratings = driver.find_elements(By.XPATH,"//span[@data-hook='rating-out-of-text']")
        for orginal_product_rating in orginal_product_ratings:
            orginal_product_rating_list.append(orginal_product_rating.text)

        orginal_product_rating_counts = driver.find_elements(By.XPATH,"//div[@data-hook='total-review-count']/span[@class='a-size-base a-color-secondary']")
        for orginal_product_rating_count in orginal_product_rating_counts:
            orginal_product_rating_count_list.append(orginal_product_rating_count.text)

        
        
        #Comments -- Sibling elements of review page are found using the parent element and data is stored to respective list
        
        review_titles = all_review.find_elements(By.XPATH,Xpath_review_title)
        for title in review_titles:
            review_title_list.append(title.text)


        all_profile_names = all_review.find_elements(By.XPATH,Xpath_reviewer_name)
        for profile_name in all_profile_names:
            reviewer_name_list.append(profile_name.text)


        all_ratings = all_review.find_elements(By.XPATH,Xpath_review_rating)
        for all_rating in all_ratings:
            review_rating_list.append(all_rating.get_attribute("innerText"))



        reviews = all_review.find_elements(By.XPATH,".//span[@class='a-size-base review-text review-text-content']")   
        for review in reviews:
            review_list.append(review.text)

            
            
        #Comments -- if there are no votes for a review a default value of zero is added to the data --       
        try:
            if len(all_review.find_elements(By.XPATH,".//span[@class='a-size-base a-color-tertiary cr-vote-text']"))>0:

                review_helpful_ratings = all_review.find_elements(By.XPATH,".//span[@class='a-size-base a-color-tertiary cr-vote-text']")   
                for review_helpful_rating in review_helpful_ratings:
                    review_helpful_rating_list.append(review_helpful_rating.text)
            else:
                review_helpful_rating_list.append("0")

        except 'no values':
            print('no value found for helful review ratings')
        

       



# In[5]:


#Comments -- A method is defined to access next review page if avilabe --

def next_page_method(driver):
    
    try:   
        while True:
            driver.refresh()
            time.sleep(3)
            next_link = driver.find_elements(By.XPATH,"//a[contains(text(), 'Next page')]")

            #Comments -- Checking if the end of review page is reached

            if len(next_link) < 1:
                review_scraper_method(driver)
                #print("No more pages left")
                break
            else: 

                #Comments -- Calling scarper method to extract data for each review page --
                
                time.sleep(3)
                review_scraper_method(driver)
                WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH,".//a[contains(text(), 'Next page')]"))).click()
                
                
    except TimeoutException:
        print ("Loading took too much time!")
        


# In[6]:


#Comments -- A method id defined to check if foreign language comments are avilable in the page --

def foreign_lang_comments_check_method():
    
    all_foreign_lang_reviews = driver.find_elements(By.XPATH,"//div[contains (@id, 'customer_review_foreign')]")  
    if len(all_foreign_lang_reviews) > 1:
        Xpath_review_title = ".//span[@data-hook='review-title']//span[1]"
        Xpath_reviewer_name = ".//div[@class='a-profile']//span[@class='a-profile-name']"
        Xpath_review_rating = ".//i[@data-hook='cmps-review-star-rating']//span[@class='a-icon-alt']"
    else:
        Xpath_review_title = ".//a[@class='a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold']/span[2]"
        Xpath_reviewer_name = ".//a[@class='a-profile']//span[@class='a-profile-name']"
        Xpath_review_rating = ".//a[@class='a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold']//span[@class='a-icon-alt']"
        
    return Xpath_review_title,Xpath_reviewer_name,Xpath_review_rating
        
        
        


# In[ ]:


#Comments -- A method is defined to create a csv file to store the extracted data --
   
def data_file_creation_method():  
   

   df = pd.DataFrame(zip(model_list,company_list,orginal_product_rating_list,orginal_product_rating_count_list,
                         review_title_list,reviewer_name_list,
                         review_rating_list,review_list,review_helpful_rating_list),columns=
                     ['Model','Company','Overall_product_rating',
                      'Overall_product_rating_count','Review_title','Reviewer_name',
                      'Review_Rating','Review','Review_helpful_rating'])

   
   
   df.to_csv('D:/amazon_review_data/Amazon_reviews.csv', mode='a', encoding='utf-8-sig',index=False, header=False)
   df.to_excel(r"D:/amazon_review_data/Amazon_reviews.xlsx",index=False)

   


# In[8]:



#Comments -- calling methods --


config_file_data = config_file_reader_method()

for x in range(1,len(config_file_data[3])):
    
    url = config_file_data[3][x]    
    driver = selenium_chrome_driver_method(url)   
    #review_scraper_method(driver)    
    next_page_method(driver)       
    print(config_file_data[2][x])
    print(len(review_title_list))



driver.quit()
data_file_creation_method()

    


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


print(len(model_list))
print(len(company_list))
print(len(orginal_product_rating_list))
print(len(orginal_product_rating_count_list))
print(len(review_title_list))
print(len(reviewer_name_list))
print(len(review_rating_list))
print(len(review_list))
print(len(review_helpful_rating_list))


# In[ ]:





# In[ ]:





# In[ ]:




