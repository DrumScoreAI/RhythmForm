#!/bin/bash
set -x

TRAINING_DATA_DIR=$1
DRYRUN=$2

cd $TRAINING_DATA_DIR

temp_m=$RANDOM.mtemp
temp_p=$RANDOM.ptemp
temp_i=$RANDOM.itemp
temp_mname=$RANDOM.mntemp
temp_pname=$RANDOM.pntemp
temp_iname=$RANDOM.intemp
temp_diff=$RANDOM.dtemp
temp_mp_diff=$RANDOM.mpdtemp

echo Finding files and comparing...
find musicxml/ -name *[0-9].xml > $temp_m
find pdfs/ -type f > $temp_p
find images/ -type f > $temp_i
for m in `cat $temp_m`; do bname=$(basename $m); name=${bname%.xml}; echo $name; done | sort > $temp_mname
for p in `cat $temp_p`; do bname=$(basename $p); name=${bname%.pdf}; echo $name; done | sort > $temp_pname
for m in `cat $temp_mname`; do if [ `grep -c ${m}_ $temp_i` -eq 0 ]; then echo $m; fi; done | sort > $temp_iname
comm -23 $temp_mname $temp_pname > $temp_mp_diff
cat $temp_mp_diff $temp_iname | sort | uniq > $temp_diff

cat $temp_diff
echo "Total musicxml files with missing pdf and/or image counterparts: `wc -l $temp_diff`"
for m_del in `cat $temp_diff`; do
    xml_path="musicxml/${m_del}.xml"
    if [ -f "$xml_path" ]; then
        if [ "$DRYRUN" != "dryrun" ]; then
            echo "$xml_path has incomplete set of files. Removing."
            rm -v "$xml_path" \
                "images/${m_del}_*.png" \
                "pdfs/${m_del}.pdf" \
                "smt/${m_del}_*.smt" \
                "musicxml/${m_del}.json" \
                "musicxml/${m_del}_altered.xml" 2>/dev/null &
        else
            echo "$xml_path has incomplete set of files. Would remove:"
            echo "  -> $xml_path"
            echo "  -> images/${m_del}_*.png (if present)"
            echo "  -> pdfs/${m_del}.pdf (if present)"
            echo "  -> smt/${m_del}_*.smt (if present)"
            echo "  -> musicxml/${m_del}.json (if present)"
            echo "  -> musicxml/${m_del}_altered.xml (if present)"
        fi
    else
        echo "$xml_path does not exist, skipping."
    fi
done

# rm -f $temp_m $temp_p $temp_i $temp_mname $temp_pname $temp_iname $temp_mp_diff $temp_mi_diff $temp_diff
wait
echo "Manually remove the following missing image files from dataset.json:"
for i in `grep image_path dataset.json | awk -F\" '{print $4}'`; do if [ ! -f "$i" ]; then echo $i does not exist; fi; done
