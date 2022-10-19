from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
from env import github_token, github_username

from nltk.tokenize.toktok import ToktokTokenizer
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import unicodedata


def get_search_csv():
    """
    A module for obtaining repo readme and language data from the github API.
    Before using this module, read through it, and follow the instructions marked
    TODO.
    After doing so, run it like this:
        python acquire.py
    To create the `data.json` file that contains the data."""

    file = 'search_results.csv'
    
    if os.path.exists(file):
        return pd.read_csv(file)
                
    else:
        good_urls = ['https://github.com/homerchen19/nba-go',
        'https://github.com/hegaojian/JetpackMvvm',
        'https://github.com/naoyashiga/Dunk',
        'https://github.com/chonyy/AI-basketball-analysis',
        'https://github.com/hegaojian/WanAndroid',
        'https://github.com/bttmly/nba',
        'https://github.com/smuyyh/SprintNBA',
        'https://github.com/linouk23/NBA-Player-Movements',
        'https://github.com/FaridSafi/react-native-basketball',
        'https://github.com/TryKickoff/kickoff',
        'https://github.com/kshvmdn/nba.js',
        'https://github.com/jaebradley/basketball_reference_web_scraper',
        'https://github.com/cwendt94/espn-api',
        'https://github.com/hegaojian/MvvmHelper',
        'https://github.com/zengm-games/zengm',
        'https://github.com/jrbadiabo/Bet-on-Sibyl',
        'https://github.com/stephanj/basketballVideoAnalysis',
        'https://github.com/KengoA/fantasy-basketball',
        'https://github.com/vishaalagartha/basketball_reference_scraper',
        'https://github.com/xwjdsh/nba-live',
        'https://github.com/neeilan/DeepPlayByPlay',
        'https://github.com/lbenz730/ncaahoopR',
        'https://github.com/RobRomijnders/RNN_basketball',
        'https://github.com/andrewgiessel/basketballcrawler',
        'https://github.com/danchyy/Basketball_Analytics',
        'https://github.com/jbkuczma/NBAreact',
        'https://github.com/rajshah4/NBA_SportVu',
        'https://github.com/gmf05/nba',
        'https://github.com/alexmonti19/dagnet',
        'https://github.com/historicalsource/nba-jam',
        'https://github.com/browlm13/Basketball-Shot-Detection',
        'https://github.com/andrewstellman/pbprdf',
        'https://github.com/adeshpande3/March-Madness-2017',
        'https://github.com/octonion/basketball',
        'https://github.com/skekre98/NBA-Search',
        'https://github.com/chonyy/basketball-shot-detection',
        'https://github.com/virajsanghvi/d3.basketball-shot-chart',
        'https://github.com/OwlTing/AI_basketball_games_video_editor',
        'https://github.com/jflancer/bigballR',
        'https://github.com/evansloan/sports.py',
        'https://github.com/dtarlow/Machine-March-Madness',
        'https://github.com/chychen/BasketballGAN',
        'https://github.com/skakac/2d-basketball-unity3d',
        'https://github.com/VamshiIITBHU14/BasketBallARKit',
        'https://github.com/AdaRoseCannon/basketball-demo',
        'https://github.com/lbenz730/NCAA_Hoops',
        'https://github.com/sportsdataverse/hoopR',
        'https://github.com/rtelmore/ballr',
        'https://github.com/lbenz730/NCAA_Hoops_Play_By_Play',
        'https://github.com/rodzam/ncaab-stats-scraper',
        'https://github.com/aoru45/LFFD-Pytorch',
        'https://github.com/octonion/basketball-m',
        'https://github.com/wcrasta/ESPN-Fantasy-Basketball',
        'https://github.com/srlesrle/betting',
        'https://github.com/kjaisingh/march-madness-2019',
        'https://github.com/ayushpai/Basketball-Detector',
        'https://github.com/BonbonLemon/basketball',
        'https://github.com/arbues6/BueStats',
        'https://github.com/leerichardson/game_simulation',
        'https://github.com/zachwill/ESPN-Basketball',
        'https://github.com/lujinzhong/Live_basketball_room',
        'https://github.com/rizkyikhwan/miracle-basketball',
        'https://github.com/sndmrc/BasketballAnalyzeR',
        'https://github.com/sportsdataverse/sportsdataverse-js',
        'https://github.com/solmos/eurolig',
        'https://github.com/zhaoyu611/basketball_trajectory_prediction',
        'https://github.com/basketballrelativity/basketball_data_science',
        'https://github.com/ed-word/Activity-Recognition',
        'https://github.com/historicalsource/nba-jam-tournament-edition',
        'https://github.com/mbjoseph/bbr',
        'https://github.com/owenauch/NBA-Fantasy-Optimizer',
        'https://github.com/cfahlgren1/Bounce',
        'https://github.com/JonnyBurger/basketball-tracker',
        'https://github.com/nguyenank/shot-plotter',
        'https://github.com/oussamabonnor1/Ball-Fall-game-Unity2D',
        'https://github.com/llimllib/ncaa-bracket-randomizer',
        'https://github.com/brettfazio/CVBallTracking',
        'https://github.com/sunkuo/joi-router',
        'https://github.com/hubsif/kodi-magentasport',
        'https://github.com/ngbede/hoop',
        'https://github.com/liang3472/BasketBall',
        'https://github.com/rukmal/Scoreboard',
        'https://github.com/imadmali/bball-hmm',
        'https://github.com/thunky-monk/kawhi',
        'https://github.com/EddM/boxscorereplay',
        'https://github.com/elwan9880/Yahoo_fantasy_basketball_analyzer',
        'https://github.com/Franpanozzo/nba-api-rest',
        'https://github.com/arbues6/Euroleague-ML',
        'https://github.com/minimaxir/ncaa-basketball',
        'https://github.com/kpascual/basketball-data-scraper',
        'https://github.com/gabarlacchi/MASK-CNN-for-actions-recognition-',
        'https://github.com/dsscollection/basketball',
        'https://github.com/gogonzo/sport',
        'https://github.com/kgilbert-cmu/basketball-gm',
        'https://github.com/nlgcat/sport_sett_basketball',
        'https://github.com/dimgold/Artificial_Curiosity',
        'https://github.com/JKH-HCA2/BasketballRecLeague',
        'https://github.com/devinmancuso/nba-start-active-players-bot',
        'https://github.com/chrisdesilva/pickup',
        'https://github.com/alfremedpal/PandasBasketball',
        'https://github.com/chenyukang/Basketball_demo',
        'https://github.com/gkaramanis/FIBA-Basketbal-World-Cup',
        'https://github.com/cxong/Dunkman',
        'https://github.com/dcampuzano101/Hoopr',
        'https://github.com/thisisevanfox/nba-my-team-ios-widget',
        'https://github.com/Innocence713/BasketballBoard',
        'https://github.com/thunderdome-data/ncaa-bracket',
        'https://github.com/elishayer/mRchmadness',
        'https://github.com/Asterisk4Magisk/Sing4Magisk',
        'https://github.com/elizabethsiegle/nba-stats-twilio-sms-bot',
        'https://github.com/Tanapruk/fb_emoji_basketball',
        'https://github.com/jbowens/nbagame',
        'https://github.com/dlm1223/march-madness',
        'https://github.com/jordanvolz/BasketballStats',
        'https://github.com/matchvs/BasketBall',
        'https://github.com/dataprofessor/basketball-heroku',
        'https://github.com/chonyy/daily-nba',
        'https://github.com/shermanash/DFSharp',
        'https://github.com/bziarkowski/euRobasket',
        'https://github.com/whsky/smarter-than-nate-silver',
        'https://github.com/embirico/basketball-object-tracker',
        'https://github.com/louis70109/PLeagueBot',
        'https://github.com/caravancodes/consumable-code-the-sport-db-api',
        'https://github.com/Basket-Analytics/BasketTracking',
        'https://github.com/jtpavlock/nbapy',
        'https://github.com/donejs/bitballs',
        'https://github.com/historicalsource/nba-hangtime',
        'https://github.com/AlexEidt/Basketball-Statistics-Tracking',
        'https://github.com/acheng1230/Web_Scraping_NBA_Data',
        'https://github.com/isovector/time2jam',
        'https://github.com/ddayto21/NBA-Time-Series-Forecasts',
        'https://github.com/kshvmdn/nba-player-tracker',
        'https://github.com/nuno-faria/HeadToHead',
        'https://github.com/jprustv/Basketball-Game',
        'https://github.com/szacho/basketball-detection',
        'https://github.com/hinkelman/Shiny-Scorekeeper',
        'https://github.com/ahnuchen/cxk-basketball',
        'https://github.com/andreweatherman/toRvik',
        'https://github.com/berrysauce/basketball',
        'https://github.com/aistairc/rotowire-modified',
        'https://github.com/SravB/NBA-Predictive-Analytics',
        'https://github.com/solmos/rfeb',
        'https://github.com/jharrilim/balldontlie-client',
        'https://github.com/LightBuzz/Kinect-Basketball-Spinner',
        'https://github.com/ramvibhakar/basketball-analytics',
        'https://github.com/jbrudvik/yahoo-fantasy-basketball',
        'https://github.com/dhatch/schneiderman',
        'https://github.com/hkair/Basketball-Action-Recognition',
        'https://github.com/3DSage/GBA-Audio-Basketball-Game',
        'https://github.com/My-Machine-Learning-Projects-CT/College-Basketball-Final-Four-Prediction',
        'https://github.com/arnav-kr/BasketBall',
        'https://github.com/Tw1ddle/samcodes-gamecircle',
        'https://github.com/scottwillson/play-by-play',
        'https://github.com/Ed-Zh/Basketball-Analytics',
        'https://github.com/jnebrera/Amateur_Basketball_Broadcasting',
        'https://github.com/yankovai/College-Basketball-Prediction',
        'https://github.com/michael-langaman/fntsylu',
        'https://github.com/magnusbakken/espn-fantasy-autopick',
        'https://github.com/kurtawirth/ncaahoopsscraper',
        'https://github.com/djblechn-su/nba-player-team-ids',
        'https://github.com/JackLich10/gamezoneR',
        'https://github.com/DavidNester/SportStreamer',
        'https://github.com/TaniaFontcuberta/Android-Basketball',
        'https://github.com/JonJonHuang/Hoops',
        'https://github.com/Esedicol/BasketballPlayerDetectection-BABPD',
        'https://github.com/DevEMCN/Kinect-Unity-Basketball',
        'https://github.com/myblackbeard/basketball-betting-bot',
        'https://github.com/seankross/bracketology',
        'https://github.com/koki25ando/NBAloveR',
        'https://github.com/lilleswing/March-Madness',
        'https://github.com/nwpu-basketball-robot/vision',
        'https://github.com/MojoJolo/fb_basketball',
        'https://github.com/EsmaShr/Ansong-Basketball',
        'https://github.com/Yao-Shao/Basketball-Game-Goal-Detector',
        'https://github.com/domkia/android-basketball-game',
        'https://github.com/johnsylvain/cbb',
        'https://github.com/christopherjenness/Similar-Shooter',
        'https://github.com/snestler/wncaahoopR',
        'https://github.com/uom-android-team2/WeBall_Statistics-Backend',
        'https://github.com/uom-android-team2/WeBall_Statistics',
        'https://github.com/timdagostino/NCAAsimulator',
        'https://github.com/rlabausa/nba-schedule-data',
        'https://github.com/yichenzhu1337/justplay',
        'https://github.com/sportsdataverse/wehoop',
        'https://github.com/kpelechrinis/adjusted_plusminus',
        'https://github.com/hqadeer/nba_scrape',
        'https://github.com/DeepSportRadar/player-reidentification-challenge',
        'https://github.com/HeroChan0330/Play-Video-With-Stm32',
        'https://github.com/treelover28/nbaMatchPredictor_PROTOTYPE',
        'https://github.com/nicidob/bbgm',
        'https://github.com/AlvinJiaozhu/Linear-Regression-Model-Basketball']

    #Create a function to collect the information and cache it as a json file
    def get_readme():
        file = 'readme_s.json'
        
        if os.path.exists(file):
            
            with open(file) as f:
            
                return json.load(f)
            
        github_info = []

        for link in url_list:
                
            github_dict = {}
            
            response = get(link)

            soup = BeautifulSoup(response.content, 'html.parser')
            
            github_dict['readme_txt'] = soup.find('div', id='readme').text
            
            list = []
            lang = soup.find_all('span', class_='color-fg-default text-bold mr-1')
            for i in lang:
                list.append(i.text)

            if list[0] == 'Jupyter Notebook':
                github_dict['language'] = "Python"
            
            else:
                github_dict['language'] = list[0]
            
            github_info.append(github_dict)

            with open(file, 'w') as f:
            
                json.dump(github_info, f)
            
        return github_info
    
    github_info = get_readme(good_urls)
    df = pd.DataFrame.from_dict(github_info)
    df.to_csv('search_results.csv')
    df = pd.read_csv('search_results.csv')
    
    return df

def basic_clean(text):
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    """
    text = text.lower()

    text = unicodedata.normalize('NFKD', text)\
                        .encode('ascii', 'ignore')\
                        .decode('utf-8', 'ignore')
    
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    return text


def tokenize(text):
    """ 
    Purpose:
        takes a string and tokenizes all words in t
    ---
    Parameters:
        
    ---
    Returns:
    
    """ 
    tokenizer = ToktokTokenizer()

    text = tokenizer.tokenize(text, return_str=True)

    return text 

def stem(text):
    """ 
    Purpose:
        to apply stemming to input text
    ---
    Parameters:
        text: the text to be stemmed
    ---
    Returns:
        text: text that has had stemming applied to it
    """

    #create the nltk stemmer object
    ps = PorterStemmer()    

    stems = [ps.stem(word) for word in text.split()]
    text = ' '.join(stems)

    return text

def lemmatize(text):
    """ 
    Purpose:
        applies lemmatization to input text 
    ---
    Parameters:
        text: the text to be lemmatized
    ---
    Returns:
        text: text that has been lemmatized
    """
    #create lemmatize object
    wnl = WordNetLemmatizer()

    lemmas = [wnl.lemmatize(word) for word in text.split()]
    text = ' '.join(lemmas)

    return text

def remove_stopwords(text, extra_words=None, exclude_words=None):
    """ 
    Purpose:
        to remove stopwords from input text 
    ---
    Parameters:
        text: text from which to remove stop words
    ---
    Returns:
        text: text that has had stopwords removed
    """

    stopwords_list = stopwords.words('english')

    if extra_words != None:
        stopwords_list.extend(extra_words)

    if exclude_words != None:
        for w in exclude_words:
            stopwords_list.remove(w)

    words = text.split()

    filtered_words = [w for w in words if w not in stopwords_list]

    # print()
    # print('Removed {} stopwords'.format(len(words) - len(filtered_words)))
    # print('---')

    text = ' '.join(filtered_words)

    return text

def clean(text, extra_words=None, exclude_words=None):
    """ 
    Purpose:
        performs basic clean, tokenization, and removal of stopwords on input text
    ---
    Parameters:
        text
        extra_words
        exclude_words
    ---
    Returns:
        text
    """

    text = basic_clean(text)
    
    text = tokenize(text)

    text = remove_stopwords(text, extra_words, exclude_words)

    return text

def flatten_languages(language):

    keepers = ['Python', 'R', 'JavaScript', 'HTML']
    
    if language not in keepers:
        language = 'other'

    return language

def prep_text(df):
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    
    """
    
    df['readme_txt'] = df['readme_txt'].str.replace(r'<[^<>]*>', '', regex=True)

    df = df.dropna() 
    df.drop([1, 4, 6, 13, 19, 60, 80, 105, 114, 136, 170, 187],inplace=True)
    df = df.drop(columns='Unnamed: 0')
    
    df.language = df.language.apply(flatten_languages)

    df['clean'] = df.readme_txt.apply(clean)
    df['stemmed'] = df.clean.apply(stem)
    df['lemmatized'] = df.clean.apply(lemmatize)

    df = df[['language', 'lemmatized']]
    df = df.replace("'", '', regex=True)
    for i in df.index:
        df.loc[i, 'word_count'] = len([word for word in df.loc[i, 'lemmatized'].split()])

    return df