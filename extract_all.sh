INPUT=/Volumes/Archive/download/twitter201701/01

find "${INPUT}" -iname '*.bz2' -print0 | while IFS= read -r -d $'\0' line; do
  echo "Processing $line"
  bzip2 -dc "$line" | python3 extract.py >> extracted.list
done
