#!/bin/sh
sed "s|SLACK_WEBHOOK_URL_PLACEHOLDER|${SLACK_WEBHOOK_URL}|g" \
    /etc/grafana/provisioning/alerting/contact-points.yml.template \
    > /etc/grafana/provisioning/alerting/contact-points.yml
exec /run.sh
