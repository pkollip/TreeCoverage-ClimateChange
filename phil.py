from bs4 import BeautifulSoup
import pandas as pd
import requests
r = requests.get("https://www.statsamerica.org/sip/rank_list.aspx?rank_label=pop1")
soup = BeautifulSoup(r.content, 'html.parser')

def main():
    print(soup.find('table', {'class': 'table table-bordered'}))

if __name__ == "__main__":
    main()