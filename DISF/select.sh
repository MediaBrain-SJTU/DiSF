file_names=(
  "chun1"
  "chun2"
  "chun3"
  "chun4"
  "chun5"
  "chun6"
  "chun7"
  "chun8"
  "chun9"
  "chun10"
)

for FILE_NAME in "${file_names[@]}"
do
  echo "selecting $FILE_NAME"
  python select_disf.py --file_name $FILE_NAME
done

echo "All files have been selected"
