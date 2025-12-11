# PageRank Dockerized Project

## Опис проєкту
Цей проєкт реалізує алгоритм PageRank на Python.
Весь код та залежності упаковані в Docker-контейнер для легкого запуску на будь-якому комп’ютері без додаткових налаштувань.

---

## Вимоги
- Встановлений Docker: [інструкції з установки](https://docs.docker.com/get-docker/)

---

## Як запустити

1. Клонуйте репозиторій та перейдіть у папку з проєктом:

    git clone https://github.com/username/repository.git
    cd repository/project

2. Збудуйте Docker-образ:

    docker build -t pagerank-app .

3. Запустіть контейнер:

    docker run pagerank-app

4. Ви побачите вивід роботи алгоритму у консолі.
