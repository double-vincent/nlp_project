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
    Purpose:
        A module for obtaining repo readme and language data from the github API.
        Before using this module, read through it, and follow the instructions marked
    TODO.
        After doing so, run it like this:
        python wrangle.py
        To create the `data.json` file that contains the data
    ---
    Parameters:
        
    ---
    Returns:
        df: dataframe with acquired data
    """

    file = 'search_results.csv'
    
    if os.path.exists(file):
        return pd.read_csv(file)
                
    else:
    

        # TODO: Make a github personal access token.
        #     1. Go here and generate a personal access token: https://github.com/settings/tokens
        #        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
        #     2. Save it in your env.py file under the variable `github_token`
        # TODO: Add your github username to your env.py file under the variable `github_username`
        # TODO: Add more repositories to the `REPOS` list below.

        REPOS = ['homerchen19/nba-go',
        'hegaojian/JetpackMvvm',
        'naoyashiga/Dunk',
        'chonyy/AI-basketball-analysis',
        'hegaojian/WanAndroid',
        'bttmly/nba',
        'smuyyh/SprintNBA',
        'linouk23/NBA-Player-Movements',
        'FaridSafi/react-native-basketball',
        'TryKickoff/kickoff',
        'kshvmdn/nba.js',
        'jaebradley/basketball_reference_web_scraper',
        'cwendt94/espn-api',
        'hegaojian/MvvmHelper',
        'zengm-games/zengm',
        'jrbadiabo/Bet-on-Sibyl',
        'stephanj/basketballVideoAnalysis',
        'KengoA/fantasy-basketball',
        'vishaalagartha/basketball_reference_scraper',
        'neilmj/BasketballData',
        'xwjdsh/nba-live',
        'neeilan/DeepPlayByPlay',
        'lbenz730/ncaahoopR',
        'RobRomijnders/RNN_basketball',
        'andrewgiessel/basketballcrawler',
        'simonefrancia/SpaceJam',
        'danchyy/Basketball_Analytics',
        'alexnoob/BasketBall-GM-Rosters',
        'jbkuczma/NBAreact',
        'FranGoitia/basketball_reference',
        'rajshah4/NBA_SportVu',
        'gmf05/nba',
        'FranGoitia/basketball-analytics',
        'alexmonti19/dagnet',
        'historicalsource/nba-jam',
        'browlm13/Basketball-Shot-Detection',
        'andrewstellman/pbprdf',
        'adeshpande3/March-Madness-2017',
        'octonion/basketball',
        'skekre98/NBA-Search',
        'chonyy/basketball-shot-detection',
        'virajsanghvi/d3.basketball-shot-chart',
        'OwlTing/AI_basketball_games_video_editor',
        'jflancer/bigballR',
        'evansloan/sports.py',
        'dtarlow/Machine-March-Madness',
        'chychen/BasketballGAN',
        'skakac/2d-basketball-unity3d',
        'VamshiIITBHU14/BasketBallARKit',
        'AdaRoseCannon/basketball-demo',
        'lbenz730/NCAA_Hoops',
        'sportsdataverse/hoopR',
        'rtelmore/ballr',
        'lbenz730/NCAA_Hoops_Play_By_Play',
        'rodzam/ncaab-stats-scraper',
        'aoru45/LFFD-Pytorch',
        'octonion/basketball-m',
        'wcrasta/ESPN-Fantasy-Basketball',
        'srlesrle/betting','kjaisingh/march-madness-2019',
        'ayushpai/Basketball-Detector',
        'BonbonLemon/basketball',
        'arbues6/BueStats',
        'fivethirtyeight/nba-player-advanced-metrics',
        'leerichardson/game_simulation',
        'zachwill/ESPN-Basketball',
        'lujinzhong/Live_basketball_room',
        'rizkyikhwan/miracle-basketball',
        'sndmrc/BasketballAnalyzeR',
        'sportsdataverse/sportsdataverse-js',
        'solmos/eurolig',
        'zhaoyu611/basketball_trajectory_prediction',
        'basketballrelativity/basketball_data_science',
        'ed-word/Activity-Recognition',
        'historicalsource/nba-jam-tournament-edition',
        'mbjoseph/bbr',
        'owenauch/NBA-Fantasy-Optimizer',
        'cfahlgren1/Bounce',
        'JonnyBurger/basketball-tracker',
        'nguyenank/shot-plotter',
        'oussamabonnor1/Ball-Fall-game-Unity2D',
        'llimllib/ncaa-bracket-randomizer',
        'brettfazio/CVBallTracking',
        'sunkuo/joi-router',
        'hubsif/kodi-magentasport',
        'ngbede/hoop',
        'liang3472/BasketBall',
        'rukmal/Scoreboard',
        'imadmali/bball-hmm',
            'thunky-monk/kawhi',
        'EddM/boxscorereplay',
        'elwan9880/Yahoo_fantasy_basketball_analyzer',
        'Franpanozzo/nba-api-rest',
        'arbues6/Euroleague-ML',
        'minimaxir/ncaa-basketball',
        'kpascual/basketball-data-scraper',
        'gabarlacchi/MASK-CNN-for-actions-recognition-',
        'dsscollection/basketball',
        'gogonzo/sport',
        'kgilbert-cmu/basketball-gm',
        'nlgcat/sport_sett_basketball',
        'dimgold/Artificial_Curiosity',
        'JKH-HCA2/BasketballRecLeague',
        'devinmancuso/nba-start-active-players-bot',
        'cryptopunksnotdead/punks.bodies',
        'tutsplus/BasketballFreeThrowUnity',
        'chrisdesilva/pickup',
        'alfremedpal/PandasBasketball',
        'chenyukang/Basketball_demo']

        headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

        if headers["Authorization"] == "token " or headers["User-Agent"] == "":
            raise Exception(
                "You need to follow the instructions marked TODO in this script before trying to use it"
            )


        def github_api_request(url: str) -> Union[List, Dict]:
            response = requests.get(url, headers=headers)
            response_data = response.json()
            if response.status_code != 200:
                raise Exception(
                    f"Error response from github api! status code: {response.status_code}, "
                    f"response: {json.dumps(response_data)}"
                )
            return response_data


        def get_repo_language(repo: str) -> str:
            url = f"https://api.github.com/repos/{repo}"
            repo_info = github_api_request(url)
            if type(repo_info) is dict:
                repo_info = cast(Dict, repo_info)
                if "language" not in repo_info:
                    raise Exception(
                        "'language' key not round in response\n{}".format(json.dumps(repo_info))
                    )
                return repo_info["language"]
            raise Exception(
                f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
            )


        def get_repo_contents(repo: str) -> List[Dict[str, str]]:
            url = f"https://api.github.com/repos/{repo}/contents/"
            contents = github_api_request(url)
            if type(contents) is list:
                contents = cast(List, contents)
                return contents
            raise Exception(
                f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
            )


        def get_readme_download_url(files: List[Dict[str, str]]) -> str:
            """
            Takes in a response from the github api that lists the files in a repo and
            returns the url that can be used to download the repo's README file.
            """
            for file in files:
                if file["name"].lower().startswith("readme"):
                    return file["download_url"]
            return ""


        def process_repo(repo: str) -> Dict[str, str]:
            """
            Takes a repo name like "gocodeup/codeup-setup-script" and returns a
            dictionary with the language of the repo and the readme contents.
            """
            contents = get_repo_contents(repo)
            readme_download_url = get_readme_download_url(contents)
            if readme_download_url == "":
                readme_contents = ""
            else:
                readme_contents = requests.get(readme_download_url).text
            return {
                "repo": repo,
                "language": get_repo_language(repo),
                "readme_contents": readme_contents,
            }


        def scrape_github_data() -> List[Dict[str, str]]:
            """
            Loop through all of the repos and process them. Returns the processed data.
            """
            return [process_repo(repo) for repo in REPOS]


        if __name__ == "__main__":
            data = scrape_github_data()
            json.dump(data, open("data.json", "w"), indent=1)
        
        df = pd.DataFrame.from_dict(data)
    
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