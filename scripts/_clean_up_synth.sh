#!/bin/bash

if [ -z $1 ] || [ -z $2 ]
then
        exit 1
fi
cd $1
xml=$2
dryrun=$3

bname=$(basename $xml)

name=${bname%.xml}
image=`ls images/$name*.png 2>/dev/null | awk '{print $1}'`
images=`ls images/$name*.png 2>/dev/null`
pdf=`ls pdfs/$name.pdf 2>/dev/null`
smts=`ls smt/$name*.smt 2>/dev/null`
if [ ! $pdf ] || [ ! $image ]
then
        if [ "$dryrun" != "dryrun" ]; then
            rm $xml 2>/dev/null
            rm $images 2>/dev/null
            rm $pdf 2>/dev/null
            rm $smts 2>/dev/null
            rm musicxml/$name.json 2>/dev/null
            rm musicxml/${name}_altered.xml 2>/dev/null
        else
            echo $bname has no pdf or image
            echo "Dry run mode - not deleting files."
            echo "Would remove: $xml"
            echo "Would remove: $images"
            echo "Would remove: $pdf"
            echo "Would remove: $smts"
            echo "Would remove: musicxml/$name.json if present"
            echo "Would remove: musicxml/${name}_altered.xml if present"
        fi
fi