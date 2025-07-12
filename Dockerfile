FROM python:3.9-slim

# Instalacja JDK 17
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean

# You may change the version of JDK if needed
# For example, to use arm64 architecture instead of amd64:
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

RUN pip3 install --no-cache-dir streamlit pyspark numpy pandas

WORKDIR /app

COPY notebooks /app

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
