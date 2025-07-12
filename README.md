# INSTRUCTION FOR DOCKER TO WORK (HOPEFULLY)

## 1. Clone repo
`git clone https://github.com/fijalkoa/bigdata.git`

## 2. Create docker containers
`docker-compose up -d`

## 3. Copy data files to namenode container
`docker cp ml-32m namenode:/`
`docker cp title.basics.tsv namenode:/`
`docker cp title.ratings.tsv namenode:/`

## 4. Move data files to hdfs
`docker exec -it namenode bash`

Inside the container:
`hdfs dfs -mkdir -p /data`
`hdfs dfs -mkdir -p /output`
`hdfs dfs -chmod 777 /output`
`hdfs dfs -put ml-32m /data/`
`hdfs dfs -put title.basics.tsv /data/`
`hdfs dfs -put title.ratings.tsv /data/`
`exit`

## 5. Get access to pyspark notebooks
`docker logs pyspark`
In logs look for sth like: "127.0.0.1:8888/token=" then open the link.


## Useful links:
To check if namenode and datanode see each other: [http://localhost:9870/dfshealth.html#tab-datanode](http://localhost:9870/dfshealth.html#tab-datanode)
Notebooks (etl+ml model): [http://127.0.0.1:8888/lab/tree/work/](http://127.0.0.1:8888/lab/tree/work/)