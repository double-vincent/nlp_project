from pyexpat import features
import pandas as pd
from requests import get
from bs4 import BeautifulSoup as soupify



def get_blog_urls(base_url, header={'User-Agent': "Codeup Data Science"}):
    """ 
    Purpose:
        get the proper url for the Codeup Data Science blog
    ---
    Parameters:
       base_url: url that forms basis of web address
       header: default, dict, 'User-Agent': "Codeup Data Science"
            user agent needed to interact with website
    ---
    Returns:
        list: list of blog urls
    """
    soup = soupify(
                    get(base_url, headers=header).content,
                    features='lxml')    

    return [link['href'] for link in soup.select('a.more-link')]

def get_blog_content(base_url='https://codeup.com/blog/', header={'User-Agent': "Codeup Data Science"}):
    """ 
    Purpose:
        to acquire content from the Codeup Data Science Blog
    ---
    Parameters:
       base_url: url that forms basis of web address
       header: default, dict, 'User-Agent': "Codeup Data Science"
            user agent needed to interact with website
    ---
    Returns:
        all_blogs: list of dictionaries, with each dictionary representing one article. 
    """

    #create empty list to hold blog content
    all_blogs = []

    blog_links = get_blog_urls(base_url)

    #loop over each blog and append title/content to list as a dictionary
    for blog in blog_links:
        blog_soup = soupify(
            get(blog, 
            headers=header).content,
            features='lxml')

        all_blogs.append(
                    {'title': blog_soup.select_one('h1.entry-title').text,
                    'original': blog_soup.select_one('div.entry-content').text.strip()})

    return pd.DataFrame(all_blogs)

def get_news_articles(fresh=False, base_url='https://inshorts.com/en/read'):
    """ 
    Purpose:
        to scarpe news articles from inshorts
    ---
    Parameters:
        fresh:
        base_url:
    ---
    Returns:

    """
    if fresh == False:
        all_articles = pd.read_csv('news_articles.csv')
        return all_articles

    #categories = get_cats(base_url)
    categories = ['business', 'sports', 'technology',  'entertainment']
    all_articles = []

    for category in categories:
        url = base_url + '/' + category
        soup = soupify(get(url).content)
        titles = [title.text for title in soup
                    .find_all('span', itemprop='headline')]
        bodies = [body.text for body in soup
                    .find_all('div', itemprop='articleBody')]
        articles = [{'title': title, 
                        'category': category,
                        'original': body} for title, body in zip(
                            titles, bodies
                        )]
        all_articles.extend(articles)
    
    all_articles = pd.DataFrame(all_articles)
    all_articles.to_csv('news_articles.csv', index=False)

    return all_articles