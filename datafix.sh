#!/bin/bash

set -x
mkdir -p /data
hadoop fs -copyToLocal /user/paulm/web-traffic-time-series-forecasting/* /data/.

apt-get update
apt-get install unzip

cd /data || exit
for f in *; do
	unzip "${f}"
done
