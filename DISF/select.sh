file_names=(
  "chunk1"
  "chunk2"
  "chunk3"
  "chunk4"
  "chunk5"
  "chunk6"
  "chunk7"
  "chunk8"
  "chunk9"
  "chunk10"
)

for FILE_NAME in "${file_names[@]}"
do
  echo "selecting $FILE_NAME"
  python select_disf.py --file_name $FILE_NAME
done

echo "All files have been selected"
