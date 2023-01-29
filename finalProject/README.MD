# SkillFactory-dspr-40 finalProject

- Большие файлы можно скачать тут
* https://drive.google.com/file/d/16EQAkXXee39yju2XW7T7ZeHayBbNK-wY/view?usp=share_link

- Докер с приложением можно запустить:
docker pull negativegreep/final_project
docker run -it --rm --name=server -p=5000:5000 negativegreep/final_project
- Протестировать точность
python test/test.py
- Ввести свои данные и получить предикт
python custom/custom.py -b 2 -yb 2010 -c 1 -ls 10000 -sf 2500 -sr [3,4,5] -city houston -st tx -zc 77068
'-b', type=int, default=2, help='number of bedrooms.'
'-yb', type=int, help='year of built.'
'-c', type=int, help='cooling, 0 - no, 1 - yes.'
'-ls', type=float, default=0, help='lotsize, sqft.'
'-sf', type=float, help='sqft.'
'-sr', type=str, help='near schools ratings in [].'
'-city', type=str, help='city.'
'-st', type=str, help='state in short form.'
'-zc', type=str, help='zipcode.'
