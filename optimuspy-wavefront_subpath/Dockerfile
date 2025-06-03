FROM cooliron/optimuspy:base

RUN apt-get update && apt-get install -y libomp-dev

# Set user and group
ARG user=optimuspy
ARG group=optimuspy
ARG uid=1000
ARG gid=1000
RUN groupadd -g ${gid} ${group}
RUN useradd -u ${uid} -g ${group} -s /bin/sh -m ${user}

RUN mkdir optimuspy

WORKDIR /optimuspy
COPY ./opsc-bin /optimuspy/opsc-bin
COPY ./opsc /optimuspy/opsc
COPY ./catch2 /optimuspy/catch2

WORKDIR /optimuspy/opsc-bin
RUN cat opsc.tar.xz | tar xJ
RUN mv lib/* /usr/local/lib/
RUN mv opsc /optimuspy/opsc
RUN rm opsc.tar.xz
RUN rmdir lib

WORKDIR /optimuspy/opsc
RUN chmod +x opsc
RUN ldconfig

WORKDIR /optimuspy/catch2
RUN wget https://github.com/catchorg/Catch2/releases/download/v3.3.2/catch_amalgamated.cpp
RUN wget https://github.com/catchorg/Catch2/releases/download/v3.3.2/catch_amalgamated.hpp
RUN make gpp
RUN make clang

WORKDIR /optimuspy
COPY ./requirements /optimuspy/requirements
RUN pip3 install -r requirements/prod.txt

COPY ./optimuspy /optimuspy/optimuspy
COPY ./scripts /optimuspy/scripts

COPY ./static /optimuspy/static
COPY ./web /optimuspy/web
COPY ./.env /optimuspy/.env
COPY ./manage.py /optimuspy/manage.py

RUN python3 manage.py collectstatic

WORKDIR /optimuspy

RUN chown -R optimuspy:optimuspy .

# Switch to user
USER ${uid}:${gid}

RUN mkdir -p tasks
