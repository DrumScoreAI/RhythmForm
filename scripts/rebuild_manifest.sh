#!/bin/bash
TRAINING_DATA_DIR=$PWD # don't use environment variable as differs from image - assume running from /app/training_data mount point
temp1=$RANDOM.tmp
i=0
CSI="\033[0K"
find $TRAINING_DATA_DIR/pdfs -name "*.pdf" > $temp1
# Use find to supply loop
find "$TRAINING_DATA_DIR/musicxml/" -name "*[0-9].xml" | while read -r xml
do
        i=$(( i + 1 ))
        xml_bn=$(basename "$xml")
        pdf="${xml_bn%.xml}.pdf"
        grep -q "$pdf" "$temp1" 2>/dev/null
        if [ $? -eq 0 ]
        then
                sed -i "s/$pdf,$xml_bn,do,n/$pdf,$xml_bn,do,p/g" $TRAINING_DATA_DIR/training_data.csv
        fi
        echo -ne "${i}${CSI}\r"
        # uncomment grep and comment echo for full monitoring
        #grep "$pdf,$xml_bn,do" $TRAINING_DATA_DIR/training_data.csv
done
echo ""
rm $temp1
echo "Done."