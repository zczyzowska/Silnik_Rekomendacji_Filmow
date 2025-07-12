# INSTRUCTION FOR DOCKER TO WORK (HOPEFULLY)

## 1. Clone repo
`git clone https://github.com/zczyzowska/Silnik_Rekomendacji_Filmow.git`

## 2. Download datasets from IMDB and MovieLens, then copy them to your folder
[IMDB dataset](https://datasets.imdbws.com)

[MovieLens dataset](https://grouplens.org/datasets/movielens/32m/)

## 3. Create docker containers
`docker-compose up -d`

## 4. Copy data files to namenode container
`docker cp ml-32m namenode:/`
`docker cp title.basics.tsv namenode:/`
`docker cp title.ratings.tsv namenode:/`

## 5. Move data files to hdfs
`docker exec -it namenode bash`

Inside the container:
`hdfs dfs -mkdir -p /data`
`hdfs dfs -mkdir -p /output`
`hdfs dfs -chmod 777 /output`
`hdfs dfs -put ml-32m /data/`
`hdfs dfs -put title.basics.tsv /data/`
`hdfs dfs -put title.ratings.tsv /data/`
`exit`

## 6. Get access to pyspark notebooks
`docker logs pyspark`
In logs look for sth like: "127.0.0.1:8888/token=" then open the link.

## 7. How does it work
The whole engine is in the `etl.ipynb` file in the `notebooks` folder. After you get through it you can run the streamlit app in`app.py`, to test how it works. Additionally, you have more notebooks, to add reddit sentiment to your engine. Have fun!

## Useful links:
To check if namenode and datanode see each other: [http://localhost:9870/dfshealth.html#tab-datanode](http://localhost:9870/dfshealth.html#tab-datanode)
Notebooks (etl+ml model): [http://127.0.0.1:8888/lab/tree/work/](http://127.0.0.1:8888/lab/tree/work/)

If your streamlit app doesn't run, you may need to change the opend-jdk version in the Dockerfile. 