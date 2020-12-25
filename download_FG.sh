# Download datasets
# FGL1_BGL1
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1X-xwyypLTm9vr-boLYPIPhGxcYaPHSNF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1X-xwyypLTm9vr-boLYPIPhGxcYaPHSNF" -O OccludedPASCAL3D_FGL1_BGL1.zip &&
# FGL2_BGL2
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dNP8YE3RJ9Pzr_jQ11O6f6eYgYnq9ROp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dNP8YE3RJ9Pzr_jQ11O6f6eYgYnq9ROp" -O OccludedPASCAL3D_FGL2_BGL2.zip &&
# FGL3_BGL3
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GsHCyAYnqcJsAgiih1vKpDQxzF3ouFxS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GsHCyAYnqcJsAgiih1vKpDQxzF3ouFxS" -O OccludedPASCAL3D_FGL3_BGL3.zip &&

# Unzip files
unzip OccludedPASCAL3D_FGL1_BGL1.zip
unzip OccludedPASCAL3D_FGL2_BGL2.zip
unzip OccludedPASCAL3D_FGL3_BGL3.zip

# Delete zipped files
rm OccludedPASCAL3D_FGL1_BGL1.zip
rm OccludedPASCAL3D_FGL2_BGL2.zip
rm OccludedPASCAL3D_FGL3_BGL3.zip

# Clear up
rm -rf /tmp/cookies.txt

