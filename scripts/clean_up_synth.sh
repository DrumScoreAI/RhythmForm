#!/bin/bash

for xml in `find musicxml/ -name *[0-9].xml`; do bname=$(basename $xml); name=${bname%.xml}; image=`ls images/$name.png 2>/dev/null`; pdf=`ls p
dfs/$name.pdf 2>/dev/null`; if [ -z $pdf ] || [ -z $image ]; then echo $bname has no pdf or image; rm -vf $xml; rm -vf $image; rm -vf $pdf; rm -vf musicxml/$name.json; rm -vf musicxml/${name}_altered.xml; fi;
done

for i in `grep image_path dataset.json | awk -F\" '{print $4}'`; do exists=`ls $i`; if [ -z $exists ]; then echo $i does not exist; fi; done
