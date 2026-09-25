# Airflow image for this project: slim base + Postgres driver + our Python dependencies.
# Dependencies are baked in at build time (not installed on container start),
# so containers start fast and every run uses the same versions.
ARG AIRFLOW_VERSION=3.3.2
ARG PYTHON_VERSION=3.12

FROM apache/airflow:slim-${AIRFLOW_VERSION}-python${PYTHON_VERSION}

ARG AIRFLOW_VERSION
ARG PYTHON_VERSION

COPY requirements.txt /tmp/requirements.txt

# Pinning apache-airflow in the same command stops pip from up/downgrading it,
# and the constraints file pins every other package to versions Airflow tested.
RUN pip install --no-cache-dir \
      "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
      -r /tmp/requirements.txt \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
