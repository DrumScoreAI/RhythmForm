#!/bin/bash

if [ -z $1 ] || [ -z $2 ]
then
        exit 1
fi
cd $1
xml=$2

bname=$(basename $xml)
name=${bname%.xml}
image=`ls images/$name.png 2>/dev/null`
pdf=`ls pdfs/$name.pdf 2>/dev/null`
if [ -z $pdf ] || [ -z $image ]
then
        echo $bname has no pdf or image
        rm -v $xml
        rm -v $image
        rm -v $pdf
        rm -v musicxml/$name.json
        rm -v musicxml/${name}_altered.xml
fi