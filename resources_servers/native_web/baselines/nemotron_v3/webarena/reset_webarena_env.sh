#!/usr/bin/env bash
set -uo pipefail

stop_after_error() {
  echo "Script stopped."
  return 1
}

# Please edit these variables to match your environment
RESET_HOST="${RESET_HOST:-${WA_HOSTNAME:-18.116.12.228}}"
RESET_TOKEN_FILE="${RESET_TOKEN_FILE:-/lustre/fsw/portfolios/nvr/users/mingjiel/root/reset_token}"
RESET_TLS_CA_CERT="${RESET_TLS_CA_CERT:-/lustre/fsw/portfolios/nvr/users/mingjiel/root/reset.crt}"

RESET_PORT="${RESET_PORT:-7777}"
RESET_SCHEME="${RESET_SCHEME:-https}"

echo "RESET_HOST: $RESET_HOST"
echo "RESET_TOKEN_FILE: $RESET_TOKEN_FILE"
echo "RESET_TLS_CA_CERT: $RESET_TLS_CA_CERT"
echo "RESET_PORT: $RESET_PORT"
echo "RESET_SCHEME: $RESET_SCHEME"
echo "RESET_ENDPOINT: $RESET_ENDPOINT"
echo "STATUS_ENDPOINT: $STATUS_ENDPOINT"
echo "CURL_ARGS: ${CURL_ARGS[*]}"

if [[ ! -s "$RESET_TOKEN_FILE" ]]; then
  echo "Missing reset bearer token file: $RESET_TOKEN_FILE" >&2
  stop_after_error || return 1 2>/dev/null || exit 1
fi

if ! RESET_TOKEN="$(<"$RESET_TOKEN_FILE")"; then
  echo "Failed to read reset bearer token file: $RESET_TOKEN_FILE" >&2
  stop_after_error || return 1 2>/dev/null || exit 1
fi

HOST="${RESET_SCHEME}://${RESET_HOST}:${RESET_PORT}"
RESET_ENDPOINT="$HOST/reset"
STATUS_ENDPOINT="$HOST/status"
CURL_ARGS=(-sS -H "Authorization: Bearer ${RESET_TOKEN}")

if [[ -n "$RESET_TLS_CA_CERT" ]]; then
  if [[ ! -r "$RESET_TLS_CA_CERT" ]]; then
    echo "Missing reset TLS CA certificate: $RESET_TLS_CA_CERT" >&2
    stop_after_error || return 1 2>/dev/null || exit 1
  fi
  CURL_ARGS+=(--cacert "$RESET_TLS_CA_CERT")
fi

# Trigger reset
echo "Triggering reset..."
if ! reset_response=$(curl "${CURL_ARGS[@]}" -o /dev/null -w "%{http_code}" "$RESET_ENDPOINT"); then
  echo "Reset trigger request ended with a curl error."
  echo "The server may still have received it; checking status..."
  reset_response=""
fi

if [[ -z "$reset_response" ]]; then
  sleep 10
elif [[ "$reset_response" == "202" ]]; then
  echo "Reset already in progress; the current reset request will not be run."
  stop_after_error || return 1 2>/dev/null || exit 1
elif [[ "$reset_response" != "200" ]]; then
  echo "Failed to trigger reset (HTTP $reset_response)"
  stop_after_error || return 1 2>/dev/null || exit 1
else
  echo "Reset triggered. Waiting for completion..."
  sleep 10
fi

# Poll status
while true; do
  if ! response=$(curl "${CURL_ARGS[@]}" -w "\n%{http_code}" "$STATUS_ENDPOINT"); then
    echo "Failed to check reset status."
    stop_after_error || return 1 2>/dev/null || exit 1
  fi

  body="${response%$'\n'*}"
  code="${response##*$'\n'}"

  if [[ "$code" == "200" && "$body" == *"Ready for duty!"* ]]; then
    echo "Reset completed successfully!"
    break
  elif [[ "$code" == "200" && "$body" == *"Reset ongoing"* ]]; then
    echo "Still resetting..."
  elif [[ "$code" == "500" ]]; then
    echo "Reset failed:"
    echo "$body"
    stop_after_error || return 1 2>/dev/null || exit 1
  else
    echo "Unexpected response (HTTP $code): $body"
    stop_after_error || return 1 2>/dev/null || exit 1
  fi

  sleep 30
done

echo "Script finished."
