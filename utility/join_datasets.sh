#!/usr/bin/env bash
# TOR
echo 'WORKIN ON TOR - AUDIO-STREAMING ... ';
cat datasets/raw/tor/AUDIO-STREAMING/AUDIO-STREAMING_COMPLETE.csv datasets/raw/nonTor/AUDIO-STREAMING/AUDIO-STREAMING_COMPLETE.csv > datasets/raw/mixed/AUDIO-STREAMING.csv
echo 'WORKIN ON TOR - BROWSING ... ';
cat datasets/raw/tor/BROWSING/BROWSING_COMPLETE.csv datasets/raw/nonTor/BROWSING/BROWSING_COMPLETE.csv > datasets/raw/mixed/BROWSING.csv
echo 'WORKIN ON TOR - CHAT ... ';
cat datasets/raw/tor/CHAT/CHAT_COMPLETE.csv datasets/raw/nonTor/CHAT/CHAT_COMPLETE.csv > datasets/raw/mixed/CHAT.csv
echo 'WORKIN ON TOR - EMAIL ... ';
cat datasets/raw/tor/EMAIL/EMAIL_COMPLETE.csv datasets/raw/nonTor/EMAIL/EMAIL_COMPLETE.csv > datasets/raw/mixed/EMAIL.csv
echo 'WORKIN ON TOR - FILE-TRANSFER ... ';
cat datasets/raw/tor/FILE-TRANSFER/FILE-TRANSFER_COMPLETE.csv datasets/raw/nonTor/FILE-TRANSFER/FILE-TRANSFER_COMPLETE.csv > datasets/raw/mixed/FILE-TRANSFER.csv
echo 'WORKIN ON TOR - P2P ... ';
cat datasets/raw/tor/P2P/P2P_COMPLETE.csv datasets/raw/nonTor/P2P/P2P_COMPLETE.csv > datasets/raw/mixed/P2P.csv
echo 'WORKIN ON TOR - VIDEO-STREAMING ... ';
cat datasets/raw/tor/VIDEO-STREAMING/VIDEO-STREAMING_COMPLETE.csv datasets/raw/nonTor/VIDEO-STREAMING/VIDEO-STREAMING_COMPLETE.csv > datasets/raw/mixed/VIDEO-STREAMING.csv
echo 'WORKIN ON TOR - VOIP ... ';
cat datasets/raw/tor/VOIP/VOIP_COMPLETE.csv datasets/raw/nonTor/VOIP/VOIP_COMPLETE.csv > datasets/raw/mixed/VOIP.csv


