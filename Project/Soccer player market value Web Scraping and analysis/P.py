import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

number = []
name = []
position = []
age = []
nation = []
team = []
value = []

for i in range(1, 2 + 1):  # n 은 페이지 수
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
    url = f'https://www.transfermarkt.com/spieler-statistik/wertvollstespieler/marktwertetop/ajax/yw1/0/page/{i}'
    r = requests.get(url, headers=headers)
    r.status_code

    soup = BeautifulSoup(r.text, 'html.parser')

    player_info = soup.find_all('tr', class_=['odd', 'even'])

    for info in player_info:
        player = info.find_all('td')

        number.append(player[0].text)
        name.append(player[3].text)
        position.append(player[4].text)
        age.append(player[5].text)
        nation.append(player[6].img['alt'])
        team.append(player[7].img['alt'])
        value.append(player[8].text.strip())

        time.sleep(1)  # 1초만 휴식

df = pd.DataFrame(
    {'number': number,
     'name': name,
     'position': position,
     'age': age,
     'nation': nation,
     'team': team,
     'value': value}
)

df.to_csv('transfermakt50.csv', index=False)