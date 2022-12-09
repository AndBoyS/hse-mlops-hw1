# hse-mlops-hw1
HSE MLOps Homeworks

Запуск flask сервера: 

```angular2html
python main.py
```

## Первое дз (до 23.10.2021)

Реализовать API (REST либо процедуры gRPC), которое умеет:
1. Обучать ML-модель с возможностью настройки гиперпараметров. При этом гиперпараметры для разных моделей могут быть разные. Минимальное количество классов моделей доступных для обучения == 2.
2. Возвращать список доступных для обучения классов моделей
3. Возвращать предсказание конкретной модели (как следствие, система должна уметь хранить несколько обученных моделей)
4. Обучать заново и удалять уже обученные модели

Оценка
- [4 балла] Работоспособность программы - то что ее можно запустить и она выполняет задачи, перечисленные в требованиях.
- [3 балла] Корректность и адекватность программы - корректная обработка ошибок, адекватный выбор структур классов, понятная документация (docstring-и адекатные здесь обязательны)
- [2 балла] Стиль кода - соблюдение стайлгайда. Буду проверять flake8 (не все ошибки на самом деле являются таковыми, но какие можно оставить – решать вам, насколько они критичны, списка нет)
- [1 балл] Swagger – Есть документация API (Swagger) с помощью flask- restx или аналога
- [2 балла] – Реализация и REST API, и gRPC

Дополнительные нюансы
- Принимать буду ссылкой на репозиторий (гитхаб, гитлаб, etc)
- Зависимости – фиксируйте. Lock файл poetry либо requirements
- Можно будет поправить или обжаловать, на что укажу, до конца дедлайна правок
- Сами дедлайны:
- Сдача ДЗ – до 23.10.2021 23:59
- Принятие правок (по ранее присланному дз) – до 30.10.2021 23:59

## Второе дз (до 11.12.2021)

К приложению из первого дз нужно добавить:
1. Работу с базой данной Postgresql.
2. Сборку самого вашего микросервиса в docker образ (образнужно запушить в docker hub)
3. Запуск вашего сервиса и БД через docker-compose
4. [полуопционально] Запуск вашего сервиса в кубере
5. [на будущее] Написать тесты

Оценка
- [3 балла] В приложение добавлена работа с БД
- [3 балла] Получившееся приложение собрано в Docker-образ и онопубликован в DockerHub
- [3 балла] Приложение можно запустить утилитой docker-compose 
- [2 балла] Приложение запускается на Kubernetes (требуетсяприложить скрипт поднятия кластера minikube и деплоймент,либо Makefile

Дополнительные нюансы 
- Принимать буду ссылкой на репозиторий (гитхаб, гитлаб, etc)
- Советую 2, 3 пункт сделать перед первым, чтобы неразворачивать БД локально, а после 1-го поправить их принеобходимости
- Можно будет поправить или обжаловать, на что укажу, до концакурса
- Сдача ДЗ – до 11.12.2022 23:59